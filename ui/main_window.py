import os
import sys
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QProgressBar, QPushButton, QScrollArea,
    QSlider, QTabWidget, QVBoxLayout, QWidget
)

from core.engine import DRAW_MODES, Keyframe, WhiteboardEngine
from core.hand_tools import (
    HandPreset, detect_pen_tip, imread_unicode,
    load_hand_presets, remove_solid_background, trim_and_pad_hand
)
from core.media import CANVAS_RESOLUTIONS, PAPER_STYLES, MediaProvider
from core.workers import ExportThread, SceneAnalyzerThread
from ui.styles import STUDIO_QSS
from ui.widgets import (
    HandGalleryWidget, HandTipPickerWidget, IMAGE_EXTS,
    PreviewStageWidget, TimelineWidget, VIDEO_EXTS
)


class WhiteboardStudioWindow(QMainWindow):
    def __init__(self, assets_dir: str):
        super().__init__()
        self.assets_dir = assets_dir
        self.hands_dir = os.path.join(assets_dir, "hands")

        self.setWindowTitle("Whiteboard Studio Pro — Edit Video Vẽ Tay 2D")
        self.resize(1380, 840)
        self.setMinimumSize(1120, 700)
        self.setStyleSheet(STUDIO_QSS)

        self.media = MediaProvider()
        self.current_frame_idx = 0
        self.is_playing = False
        self.scenes: List[int] = []
        self.keyframes: List[Keyframe] = []
        self.plan_cache: OrderedDict[Tuple[int, int, str], dict] = OrderedDict()
        self._preview_session: dict = {}

        self.presets: List[HandPreset] = []
        self.hand_map: Dict[str, np.ndarray] = {}
        self.active_hand_id: str = "marker_vector"
        self.active_tip_x: int = 20
        self.active_tip_y: int = 69
        self.active_hand_scale: float = 0.65

        self.analyzer: Optional[SceneAnalyzerThread] = None
        self.exporter: Optional[ExportThread] = None

        self._load_preset_hands()
        self._init_timers()
        self._build_ui()
        self._bind_shortcuts()
        self._sync_hand_inspector()
        self._update_project_badge()

    def _load_preset_hands(self):
        self.presets = load_hand_presets(self.hands_dir)
        for p in self.presets:
            path = os.path.join(self.hands_dir, p.filename)
            img = imread_unicode(path, cv2.IMREAD_UNCHANGED)
            if img is not None and len(img.shape) == 3 and img.shape[2] == 4:
                self.hand_map[p.id] = img

        if self.presets:
            first = self.presets[0]
            self.active_hand_id = first.id
            self.active_tip_x = first.tip_x
            self.active_tip_y = first.tip_y
            self.active_hand_scale = first.default_scale

    def _init_timers(self):
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._on_play_tick)

        self._detail_debounce = QTimer(self)
        self._detail_debounce.setSingleShot(True)
        self._detail_debounce.timeout.connect(self.refresh_frame)

    def _bind_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, activated=self.toggle_play)
        QShortcut(QKeySequence(Qt.Key.Key_K), self, activated=self.add_keyframe_at_current)
        QShortcut(QKeySequence(Qt.Key.Key_Delete), self, activated=self.delete_selected_kf)
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, activated=lambda: self._step_frame(-1))
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, activated=lambda: self._step_frame(1))
        QShortcut(QKeySequence("Shift+Left"), self, activated=lambda: self._step_frame(-10))
        QShortcut(QKeySequence("Shift+Right"), self, activated=lambda: self._step_frame(10))

    # ==========================================================
    # XÂY DỰNG GIAO DIỆN GỌN, TRỰC QUAN (4 VÙNG CHUẨN STUDIO)
    # ==========================================================
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ----- 1. TOP TOOLBAR -----
        topbar = QFrame()
        topbar.setObjectName("TopBar")
        topbar.setFixedHeight(52)
        tb = QHBoxLayout(topbar)
        tb.setContentsMargins(14, 6, 14, 6)
        tb.setSpacing(10)

        brand = QLabel("✍️ Whiteboard Studio")
        brand.setObjectName("AppBrand")
        tb.addWidget(brand)

        btn_open_v = QPushButton("🎬 Nhập Video")
        btn_open_v.clicked.connect(self.load_video_dialog)
        btn_open_i = QPushButton("🖼 Thêm Ảnh")
        btn_open_i.setObjectName("primary")
        btn_open_i.clicked.connect(self.add_images_dialog)
        tb.addWidget(btn_open_v)
        tb.addWidget(btn_open_i)

        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.VLine)
        sep1.setStyleSheet("color: #283044;")
        tb.addWidget(sep1)

        self.btn_scan_scenes = QPushButton("🔍 Quét Cảnh")
        self.btn_scan_scenes.setToolTip("Tự động nhận diện điểm chuyển cảnh trong Video")
        self.btn_scan_scenes.clicked.connect(self.analyze_scenes)

        self.btn_auto_keys = QPushButton("⚡ Tự Gắn Key")
        self.btn_auto_keys.setToolTip("Tạo Keyframe vẽ tay tự động cho các cảnh đã quét")
        self.btn_auto_keys.clicked.connect(self.auto_apply_scene_keys)

        tb.addWidget(self.btn_scan_scenes)
        tb.addWidget(self.btn_auto_keys)

        tb.addStretch()

        self.lbl_status_pill = QLabel("Chưa có media")
        self.lbl_status_pill.setObjectName("StatusPill")
        tb.addWidget(self.lbl_status_pill)

        self.btn_top_export = QPushButton("🚀 Xuất Video MP4")
        self.btn_top_export.setObjectName("success")
        self.btn_top_export.clicked.connect(self.export_video)
        tb.addWidget(self.btn_top_export)

        outer.addWidget(topbar)

        # ----- 2. BODY: STAGE (TRÁI) + INSPECTOR (PHẢI) -----
        body = QWidget()
        b_lay = QHBoxLayout(body)
        b_lay.setContentsMargins(0, 0, 0, 0)
        b_lay.setSpacing(0)

        # Vùng Trái (Stage + Transport + Timeline)
        left_col = QWidget()
        l_lay = QVBoxLayout(left_col)
        l_lay.setContentsMargins(12, 10, 12, 8)
        l_lay.setSpacing(8)

        self.stage = PreviewStageWidget()
        self.stage.files_dropped.connect(self._handle_dropped_files)
        self.stage.request_load_video.connect(self.load_video_dialog)
        self.stage.request_load_images.connect(self.add_images_dialog)
        l_lay.addWidget(self.stage, stretch=1)

        # Thanh điều khiển phát (Transport)
        transport = QFrame()
        transport.setObjectName("TransportBar")
        transport.setFixedHeight(44)
        tr = QHBoxLayout(transport)
        tr.setContentsMargins(10, 4, 10, 4)
        tr.setSpacing(8)

        btn_prev = QPushButton("⏮")
        btn_prev.setFixedWidth(36)
        btn_prev.setToolTip("Lùi 1 frame (Phím ←)")
        btn_prev.clicked.connect(lambda: self._step_frame(-1))

        self.btn_play = QPushButton("▶ Phát")
        self.btn_play.setObjectName("primary")
        self.btn_play.setMinimumWidth(88)
        self.btn_play.setToolTip("Phát / Tạm dừng (Phím Space)")
        self.btn_play.clicked.connect(self.toggle_play)

        btn_next = QPushButton("⏭")
        btn_next.setFixedWidth(36)
        btn_next.setToolTip("Tiến 1 frame (Phím →)")
        btn_next.clicked.connect(lambda: self._step_frame(1))

        self.lbl_time = QLabel("00:00.0 / 00:00.0")
        self.lbl_time.setFont(QFont("Consolas", 11))
        self.lbl_time.setStyleSheet("color: #CBD5E1; padding: 0 6px;")

        self.chk_live_fx = QCheckBox("Xem hiệu ứng trực tiếp")
        self.chk_live_fx.setChecked(True)
        self.chk_live_fx.stateChanged.connect(self.refresh_frame)

        self.btn_add_key = QPushButton("◆ Gắn Key tại đây (K)")
        self.btn_add_key.setObjectName("accent")
        self.btn_add_key.clicked.connect(self.add_keyframe_at_current)

        tr.addWidget(btn_prev)
        tr.addWidget(self.btn_play)
        tr.addWidget(btn_next)
        tr.addWidget(self.lbl_time)
        tr.addStretch()
        tr.addWidget(self.chk_live_fx)
        tr.addWidget(self.btn_add_key)
        l_lay.addWidget(transport)

        # Timeline
        self.timeline = TimelineWidget()
        self.timeline.seek_requested.connect(self.seek_to)
        self.timeline.keyframe_selected.connect(self._on_timeline_select_kf)
        self.timeline.keyframe_moved.connect(self._on_timeline_move_kf)
        l_lay.addWidget(self.timeline)

        # Chú thích ngắn gọn dưới Timeline
        legend = QHBoxLayout()
        legend.setContentsMargins(4, 0, 4, 0)
        lbl_leg = QLabel(
            "Màu Key:  🟡 Vẽ Nét + Tô Màu   🟢 Chỉ Vẽ Nét   🟣 Phác Thảo Thực Tế   🔵 Cọ Tô Nhanh   |   "
            "Phím tắt: Space Phát/Dừng · K Gắn Key · ←/→ Chỉnh từng frame"
        )
        lbl_leg.setStyleSheet("color: #64748B; font-size: 11px;")
        legend.addWidget(lbl_leg)
        l_lay.addLayout(legend)

        # ----- VÙNG PHẢI: INSPECTOR 3 TAB GỌN ĐẸP -----
        inspector = QFrame()
        inspector.setObjectName("InspectorPanel")
        inspector.setFixedWidth(392)
        ins_lay = QVBoxLayout(inspector)
        ins_lay.setContentsMargins(8, 8, 8, 8)
        ins_lay.setSpacing(6)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_tab_effects(), "🎨 Hiệu Ứng")
        self.tabs.addTab(self._build_tab_hands(), "✋ Bàn Tay 2D")
        self.tabs.addTab(self._build_tab_export(), "⚙ Dự Án && Xuất")
        ins_lay.addWidget(self.tabs)

        b_lay.addWidget(left_col, stretch=1)
        b_lay.addWidget(inspector)
        outer.addWidget(body, stretch=1)

    def _wrap_scroll(self, widget: QWidget) -> QScrollArea:
        widget.setMaximumWidth(354)
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sa.setWidget(widget)
        return sa

    def _make_card(self, title: str) -> Tuple[QFrame, QVBoxLayout]:
        card = QFrame()
        card.setObjectName("Card")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)
        lbl = QLabel(title)
        lbl.setObjectName("SectionTitle")
        lay.addWidget(lbl)
        return card, lay

    # ---------- TAB 1: HIỆU ỨNG & KEYFRAMES ----------
    def _build_tab_effects(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(4, 6, 4, 8)
        lay.setSpacing(10)

        # Card 1: Danh sách Keyframe
        c_list, cl_lay = self._make_card("Danh Sách Keyframe")
        self.list_kf = QListWidget()
        self.list_kf.setMinimumHeight(115)
        self.list_kf.setMaximumHeight(150)
        self.list_kf.currentRowChanged.connect(self._on_list_select_kf)
        cl_lay.addWidget(self.list_kf)

        kf_btns = QHBoxLayout()
        b_add = QPushButton("+ Gắn Key")
        b_add.clicked.connect(self.add_keyframe_at_current)
        b_del = QPushButton("🗑 Xóa Key")
        b_del.setObjectName("danger")
        b_del.clicked.connect(self.delete_selected_kf)
        kf_btns.addWidget(b_add)
        kf_btns.addWidget(b_del)
        cl_lay.addLayout(kf_btns)
        lay.addWidget(c_list)

        # Card 2: Chế độ vẽ trực quan
        c_mode, cm_lay = self._make_card("Chế Độ Vẽ Tay")
        self.mode_group = QButtonGroup(self)
        self.mode_buttons: Dict[str, QPushButton] = {}
        mode_descs = {
            "Vẽ Nét + Tô Màu": "🟡  Vẽ Nét + Tô Màu (Khuyên dùng)",
            "Chỉ Vẽ Nét": "🟢  Chỉ Vẽ Nét + Đánh Bóng Chì",
            "Phác Thảo Thực Tế": "🟣  Phác Thảo Thực Tế (Chì → Màu)",
            "Cọ Tô Nhanh": "🔵  Cọ Tô Nhanh (Hợp cho Video)",
        }
        mode_tips = {
            "Vẽ Nét + Tô Màu": "Phác thảo toàn bộ đường nét khung hình trước rồi tô màu từng vùng mượt mà",
            "Chỉ Vẽ Nét": "Vẽ nét chì và đánh bóng chéo nghệ thuật lên nền giấy trắng hoặc bảng phấn",
            "Phác Thảo Thực Tế": "Lướt cọ mở bản phác thảo chì rồi chuyển dần sang ảnh màu gốc",
            "Cọ Tô Nhanh": "Quét cọ lớn trực tiếp để lộ khung hình nhanh, rất hợp cho clip video thực tế",
        }
        for m in DRAW_MODES:
            btn = QPushButton(mode_descs[m])
            btn.setObjectName("modeBtn")
            btn.setCheckable(True)
            btn.setToolTip(mode_tips[m])
            btn.clicked.connect(lambda _chk, mode=m: self._on_mode_btn_clicked(mode))
            self.mode_group.addButton(btn)
            self.mode_buttons[m] = btn
            cm_lay.addWidget(btn)
        self.mode_buttons[DRAW_MODES[0]].setChecked(True)
        lay.addWidget(c_mode)

        # Card 3: Tinh chỉnh nét & chuyển động
        c_tune, ct_lay = self._make_card("Tinh Chỉnh Độ Mượt & Nét")
        form = QFormLayout()
        form.setVerticalSpacing(9)
        form.setHorizontalSpacing(10)

        self.spn_dur = QDoubleSpinBox()
        self.spn_dur.setRange(1.0, 20.0)
        self.spn_dur.setSingleStep(0.5)
        self.spn_dur.setSuffix(" giây")
        self.spn_dur.setValue(4.5)
        self.spn_dur.valueChanged.connect(self._on_kf_param_changed)
        form.addRow("Thời lượng:", self.spn_dur)

        # Slider chi tiết kèm badge số
        det_row = QHBoxLayout()
        self.sld_detail = QSlider(Qt.Orientation.Horizontal)
        self.sld_detail.setRange(15, 95)
        self.sld_detail.setValue(60)
        self.lbl_detail_val = QLabel("60%")
        self.lbl_detail_val.setObjectName("ValueBadge")
        self.sld_detail.valueChanged.connect(self._on_detail_slider_changed)
        det_row.addWidget(self.sld_detail, stretch=1)
        det_row.addWidget(self.lbl_detail_val)
        form.addRow("Độ chi tiết nét:", det_row)

        self.cmb_easing = QComboBox()
        self.cmb_easing.addItems(["Tự nhiên (Ease)", "Đều (Linear)"])
        self.cmb_easing.currentTextChanged.connect(self._on_kf_param_changed)
        form.addRow("Nhịp tay vẽ:", self.cmb_easing)
        ct_lay.addLayout(form)

        self.chk_shadow = QCheckBox("Bóng đổ mềm của bàn tay trên giấy")
        self.chk_shadow.setChecked(True)
        self.chk_shadow.stateChanged.connect(self._on_kf_param_changed)

        self.chk_tilt = QCheckBox("Nghiêng cổ tay nhẹ theo hướng nét vẽ")
        self.chk_tilt.setChecked(True)
        self.chk_tilt.stateChanged.connect(self._on_kf_param_changed)

        ct_lay.addWidget(self.chk_shadow)
        ct_lay.addWidget(self.chk_tilt)

        btn_apply_all = QPushButton("✨ Áp dụng thông số này cho TẤT CẢ Key")
        btn_apply_all.clicked.connect(self._apply_effect_to_all_kfs)
        ct_lay.addWidget(btn_apply_all)

        lay.addWidget(c_tune)
        lay.addStretch()
        return self._wrap_scroll(page)

    # ---------- TAB 2: BÀN TAY 2D TÁCH NỀN ----------
    def _build_tab_hands(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(4, 6, 4, 8)
        lay.setSpacing(10)

        # Card 1: Thư viện mẫu bàn tay 2D
        c_gal, cg_lay = self._make_card("Mẫu Bàn Tay 2D Tách Nền Sẵn")
        self.hand_gallery = HandGalleryWidget()
        self.hand_gallery.set_presets(self.presets, self.hand_map, self.active_hand_id)
        self.hand_gallery.preset_selected.connect(self._on_select_hand_preset)
        cg_lay.addWidget(self.hand_gallery)

        self.chk_hand_apply_all = QCheckBox("Áp dụng mẫu tay này cho tất cả Keyframe")
        self.chk_hand_apply_all.setChecked(True)
        cg_lay.addWidget(self.chk_hand_apply_all)

        btn_upload_hand = QPushButton("📂 Tải ảnh bàn tay riêng (PNG / JPG)")
        btn_upload_hand.setToolTip("Hỗ trợ tự động tách nền trắng/xanh nếu ảnh chưa trong suốt")
        btn_upload_hand.clicked.connect(self.load_custom_hand)
        cg_lay.addWidget(btn_upload_hand)
        lay.addWidget(c_gal)

        # Card 2: Kích thước & Căn chỉnh đầu bút trực quan
        c_cal, cc_lay = self._make_card("Kích Thước & Tâm Đầu Bút")

        sc_row = QHBoxLayout()
        lbl_sc = QLabel("Kích thước tay:")
        self.sld_scale = QSlider(Qt.Orientation.Horizontal)
        self.sld_scale.setRange(25, 160)
        self.sld_scale.setValue(int(round(self.active_hand_scale * 100)))
        self.lbl_scale_val = QLabel(f"{int(round(self.active_hand_scale * 100))}%")
        self.lbl_scale_val.setObjectName("ValueBadge")
        self.sld_scale.valueChanged.connect(self._on_scale_slider_changed)
        sc_row.addWidget(lbl_sc)
        sc_row.addWidget(self.sld_scale, stretch=1)
        sc_row.addWidget(self.lbl_scale_val)
        cc_lay.addLayout(sc_row)

        lbl_tip_hint = QLabel("Nhấp chuột lên đầu bút trên ảnh bên dưới nếu cần chỉnh lại tâm:")
        lbl_tip_hint.setObjectName("MutedLabel")
        lbl_tip_hint.setWordWrap(True)
        cc_lay.addWidget(lbl_tip_hint)

        self.tip_picker = HandTipPickerWidget()
        self.tip_picker.tip_changed.connect(self._on_tip_picked)
        cc_lay.addWidget(self.tip_picker, alignment=Qt.AlignmentFlag.AlignCenter)

        tip_foot = QHBoxLayout()
        self.lbl_tip_coords = QLabel(f"Tọa độ đầu bút: ({self.active_tip_x}, {self.active_tip_y})")
        self.lbl_tip_coords.setObjectName("MutedLabel")
        btn_auto_tip = QPushButton("🎯 Tự dò đầu bút")
        btn_auto_tip.clicked.connect(self._auto_detect_current_tip)
        tip_foot.addWidget(self.lbl_tip_coords)
        tip_foot.addStretch()
        tip_foot.addWidget(btn_auto_tip)
        cc_lay.addLayout(tip_foot)

        lay.addWidget(c_cal)
        lay.addStretch()
        return self._wrap_scroll(page)

    # ---------- TAB 3: DỰ ÁN & XUẤT VIDEO ----------
    def _build_tab_export(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(4, 6, 4, 8)
        lay.setSpacing(10)

        # Card 1: Nền giấy & Khổ khung hình
        c_can, cn_lay = self._make_card("Nền Giấy & Khổ Video (Chế độ Ảnh)")
        form_c = QFormLayout()
        form_c.setVerticalSpacing(8)

        self.cmb_paper = QComboBox()
        for pid, (label, _, _) in PAPER_STYLES.items():
            self.cmb_paper.addItem(label, userData=pid)
        self.cmb_paper.currentIndexChanged.connect(self._on_paper_or_res_changed)
        form_c.addRow("Chất liệu giấy:", self.cmb_paper)

        self.cmb_res = QComboBox()
        for rid, (label, _, _) in CANVAS_RESOLUTIONS.items():
            self.cmb_res.addItem(label, userData=rid)
        self.cmb_res.currentIndexChanged.connect(self._on_paper_or_res_changed)
        form_c.addRow("Khổ video:", self.cmb_res)
        cn_lay.addLayout(form_c)
        lay.addWidget(c_can)

        # Card 2: Quản lý Slide Ảnh
        c_slides, cs_lay = self._make_card("Danh Sách Ảnh Đã Nhập")
        self.list_slides = QListWidget()
        self.list_slides.setMinimumHeight(95)
        self.list_slides.setMaximumHeight(130)
        cs_lay.addWidget(self.list_slides)

        s_btns = QHBoxLayout()
        b_add_more = QPushButton("+ Thêm Ảnh")
        b_add_more.clicked.connect(self.add_images_dialog)
        b_del_slide = QPushButton("🗑 Xóa Ảnh Chọn")
        b_del_slide.clicked.connect(self._remove_selected_slide)
        s_btns.addWidget(b_add_more)
        s_btns.addWidget(b_del_slide)
        cs_lay.addLayout(s_btns)
        lay.addWidget(c_slides)

        # Card 3: Xuất Video MP4
        c_exp, ce_lay = self._make_card("Xuất Video MP4 (H.264)")
        form_e = QFormLayout()
        self.cmb_quality = QComboBox()
        self.cmb_quality.addItem("Chất lượng Cao (CRF 17)", userData=17)
        self.cmb_quality.addItem("Cân bằng (CRF 20 — Khuyên dùng)", userData=20)
        self.cmb_quality.addItem("File Nhẹ (CRF 24)", userData=24)
        self.cmb_quality.setCurrentIndex(1)
        form_e.addRow("Chuẩn nén:", self.cmb_quality)
        ce_lay.addLayout(form_e)

        self.chk_audio = QCheckBox("Giữ âm thanh gốc (khi chỉnh sửa Video)")
        self.chk_audio.setChecked(True)
        ce_lay.addWidget(self.chk_audio)

        exp_btns = QHBoxLayout()
        self.btn_export = QPushButton("🚀 Bắt Đầu Xuất MP4")
        self.btn_export.setObjectName("success")
        self.btn_export.clicked.connect(self.export_video)

        self.btn_cancel_export = QPushButton("Hủy")
        self.btn_cancel_export.setObjectName("danger")
        self.btn_cancel_export.setEnabled(False)
        self.btn_cancel_export.clicked.connect(self._cancel_export)

        exp_btns.addWidget(self.btn_export, stretch=1)
        exp_btns.addWidget(self.btn_cancel_export)
        ce_lay.addLayout(exp_btns)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        ce_lay.addWidget(self.progress)

        self.lbl_export_status = QLabel("Sẵn sàng xuất video.")
        self.lbl_export_status.setObjectName("MutedLabel")
        ce_lay.addWidget(self.lbl_export_status)

        lay.addWidget(c_exp)
        lay.addStretch()
        return self._wrap_scroll(page)

    # ==========================================================
    # XỬ LÝ NGUỒN MEDIA (VIDEO / ẢNH / DRAG & DROP)
    # ==========================================================
    def _handle_dropped_files(self, paths: List[str]):
        vids = [p for p in paths if os.path.splitext(p)[1].lower() in VIDEO_EXTS]
        imgs = [p for p in paths if os.path.splitext(p)[1].lower() in IMAGE_EXTS]
        if vids:
            self.load_video_file(vids[0])
        elif imgs:
            self.add_image_files(imgs)

    def load_video_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Chọn Video đầu vào", "", "Video (*.mp4 *.mov *.avi *.mkv *.webm)"
        )
        if path:
            self.load_video_file(path)

    def load_video_file(self, path: str):
        if self.is_playing:
            self.toggle_play()
        if not self.media.load_video(path):
            QMessageBox.warning(self, "Lỗi", "Không thể mở file video đã chọn.")
            return

        self.current_frame_idx = 0
        self.scenes.clear()
        self.keyframes.clear()
        self.plan_cache.clear()
        self._preview_session.clear()

        self.cmb_res.setEnabled(False)
        self._refresh_slide_list()
        self._refresh_kf_list()
        self._refresh_timeline()
        self._update_project_badge(os.path.basename(path))
        self.seek_to(0)

    def add_images_dialog(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Chọn 1 hoặc nhiều ảnh để ghép thành video vẽ tay",
            "", "Ảnh (*.png *.jpg *.jpeg *.webp *.bmp)"
        )
        if paths:
            self.add_image_files(paths)

    def add_image_files(self, paths: List[str]):
        if self.is_playing:
            self.toggle_play()

        dur_each = self.spn_dur.value()
        was_images = (self.media.source_type == "images")
        ok, added_items = self.media.add_images(paths, sec_each=dur_each)
        if not ok:
            QMessageBox.warning(self, "Lỗi", "Không đọc được các ảnh đã chọn.")
            return

        if not was_images:
            self.scenes.clear()
            self.keyframes.clear()
            self.plan_cache.clear()
            self._preview_session.clear()

        self.cmb_res.setEnabled(True)
        cur_mode = self._get_checked_mode()
        for start_f, dur_s in added_items:
            self.scenes.append(start_f)
            self.keyframes.append(Keyframe(
                frame_idx=start_f,
                duration_sec=dur_s,
                mode=cur_mode,
                detail=self.sld_detail.value(),
                hand_scale=self.active_hand_scale,
                tip_x=self.active_tip_x,
                tip_y=self.active_tip_y,
                hand_id=self.active_hand_id,
                easing=self.cmb_easing.currentText(),
                enable_shadow=self.chk_shadow.isChecked(),
                enable_tilt=self.chk_tilt.isChecked(),
            ))

        self.keyframes.sort(key=lambda k: k.frame_idx)
        self._refresh_slide_list()
        self._refresh_kf_list()
        if self.keyframes:
            self.list_kf.setCurrentRow(len(self.keyframes) - len(added_items))
        self._refresh_timeline()
        self._update_project_badge()

        # Nhảy tới ~38% của ảnh đầu tiên mới thêm để khách hàng thấy ngay bàn tay đang vẽ
        first_new = added_items[0][0]
        self.seek_to(first_new + int(round(dur_each * self.media.fps * 0.38)))

    def _remove_selected_slide(self):
        if self.media.source_type != "images":
            return
        row = self.list_slides.currentRow()
        if not (0 <= row < len(self.media.image_clips)):
            return

        self.media.image_clips.pop(row)
        self.media._rebuild_clip_ranges()
        # Đồng bộ lại keyframes theo slide còn lại
        if 0 <= row < len(self.keyframes):
            self.keyframes.pop(row)
        new_starts = self.media.get_clip_starts()
        for i, sf in enumerate(new_starts):
            if i < len(self.keyframes):
                self.keyframes[i].frame_idx = sf
        self.scenes = list(new_starts)
        self.plan_cache.clear()
        self._preview_session.clear()

        self._refresh_slide_list()
        self._refresh_kf_list()
        self._refresh_timeline()
        self._update_project_badge()
        if self.media.image_clips:
            self.seek_to(min(self.current_frame_idx, self.media.total_frames - 1))
        else:
            self.media.source_type = "none"
            self.stage.show_empty()

    def _on_paper_or_res_changed(self):
        paper_id = self.cmb_paper.currentData()
        res_id = self.cmb_res.currentData()
        self.media.set_paper_and_resolution(paper_id, res_id)
        self.plan_cache.clear()
        self._preview_session.clear()
        self._update_project_badge()
        self.refresh_frame()

    def _refresh_slide_list(self):
        self.list_slides.clear()
        for i, clip in enumerate(self.media.image_clips):
            self.list_slides.addItem(
                f"Ảnh {i + 1}: {os.path.basename(clip.path)} ({clip.duration_sec:.1f}s)"
            )

    def _update_project_badge(self, video_name: Optional[str] = None):
        if self.media.source_type == "none":
            self.lbl_status_pill.setText("Chưa có media")
            return
        total_sec = self.media.total_frames / max(1.0, self.media.fps)
        if self.media.source_type == "images":
            txt = f"{len(self.media.image_clips)} ảnh · {total_sec:.1f}s · {self.media.width}×{self.media.height}"
        else:
            name = video_name or os.path.basename(self.media.video_path or "Video")
            txt = f"{name} · {total_sec:.1f}s · {self.media.width}×{self.media.height}"
        self.lbl_status_pill.setText(txt)

    # ==========================================================
    # PHÁT & RENDER KHUNG HÌNH
    # ==========================================================
    def toggle_play(self):
        if self.media.source_type == "none":
            return
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play.setText("⏸ Tạm dừng")
            interval = max(16, int(round(1000.0 / min(60.0, max(15.0, self.media.fps)))))
            self.play_timer.start(interval)
        else:
            self.btn_play.setText("▶ Phát")
            self.play_timer.stop()

    def _on_play_tick(self):
        if self.media.source_type == "none":
            return
        if self.current_frame_idx + 1 >= self.media.total_frames:
            self.toggle_play()
            return
        self.seek_to(self.current_frame_idx + 1)

    def _step_frame(self, delta: int):
        if self.media.source_type == "none":
            return
        if self.is_playing:
            self.toggle_play()
        self.seek_to(self.current_frame_idx + delta)

    def seek_to(self, frame_idx: int):
        if self.media.source_type == "none":
            return
        self.current_frame_idx = max(0, min(self.media.total_frames - 1, frame_idx))
        self.timeline.set_current_frame(self.current_frame_idx)
        self.refresh_frame()

    def _fmt_time(self, f_idx: int) -> str:
        sec = f_idx / max(1.0, self.media.fps)
        return f"{int(sec // 60):02d}:{sec % 60:04.1f}"

    def _get_or_build_plan(self, kf: Keyframe) -> Optional[dict]:
        key = (kf.frame_idx, kf.detail, self.media.paper_style)
        if key in self.plan_cache:
            self.plan_cache.move_to_end(key)
            return self.plan_cache[key]
        base = self.media.read_frame(kf.frame_idx)
        if base is None:
            return None
        plan = WhiteboardEngine.prepare_sketch_plan(base, kf.detail, self.media.paper_style)
        self.plan_cache[key] = plan
        if len(self.plan_cache) > 24:
            self.plan_cache.popitem(last=False)
        return plan

    def refresh_frame(self):
        if self.media.source_type == "none":
            self.stage.show_empty()
            return

        frame = self.media.read_frame(self.current_frame_idx)
        if frame is None:
            return

        if self.chk_live_fx.isChecked():
            for kf in self.keyframes:
                dur_f = max(1, int(round(kf.duration_sec * self.media.fps)))
                if kf.frame_idx <= self.current_frame_idx < kf.frame_idx + dur_f:
                    prog = (self.current_frame_idx - kf.frame_idx) / max(1, dur_f - 1)
                    plan = self._get_or_build_plan(kf)
                    if plan is not None:
                        hand_bgra = self.hand_map.get(kf.hand_id)
                        if hand_bgra is None and self.presets:
                            hand_bgra = self.hand_map.get(self.presets[0].id)
                        frame = WhiteboardEngine.render_frame(
                            plan, prog, hand_bgra, kf, session_state=self._preview_session
                        )
                    break

        self.stage.show_frame(frame)
        self.lbl_time.setText(
            f"{self._fmt_time(self.current_frame_idx)} / {self._fmt_time(self.media.total_frames)}"
        )

    # ==========================================================
    # PHÂN TÍCH CẢNH & QUẢN LÝ KEYFRAME
    # ==========================================================
    def analyze_scenes(self):
        if self.media.source_type == "none":
            QMessageBox.information(self, "Thông báo", "Hãy nhập Video hoặc Ảnh trước.")
            return
        self.btn_scan_scenes.setEnabled(False)
        self.analyzer = SceneAnalyzerThread(self.media)
        self.analyzer.progress.connect(self.progress.setValue)
        self.analyzer.finished_scenes.connect(self._on_scenes_done)
        self.analyzer.start()

    def _on_scenes_done(self, scenes: list):
        self.scenes = scenes
        self.btn_scan_scenes.setEnabled(True)
        self._refresh_timeline()
        QMessageBox.information(self, "Hoàn tất", f"Đã phát hiện {len(scenes)} điểm cảnh.")

    def auto_apply_scene_keys(self):
        if not self.scenes:
            if self.media.source_type == "images":
                self.scenes = self.media.get_clip_starts()
            else:
                QMessageBox.information(self, "Thông báo", "Hãy bấm 'Quét Cảnh' trước.")
                return

        cur_mode = self._get_checked_mode()
        self.keyframes = [
            Keyframe(
                frame_idx=s,
                duration_sec=self.spn_dur.value(),
                mode=cur_mode,
                detail=self.sld_detail.value(),
                hand_scale=self.active_hand_scale,
                tip_x=self.active_tip_x,
                tip_y=self.active_tip_y,
                hand_id=self.active_hand_id,
                easing=self.cmb_easing.currentText(),
                enable_shadow=self.chk_shadow.isChecked(),
                enable_tilt=self.chk_tilt.isChecked(),
            )
            for s in self.scenes
        ]
        self._preview_session.clear()
        self._refresh_kf_list()
        if self.keyframes:
            self.list_kf.setCurrentRow(0)
        self._refresh_timeline()
        self.refresh_frame()

    def add_keyframe_at_current(self):
        if self.media.source_type == "none":
            return
        new_kf = Keyframe(
            frame_idx=self.current_frame_idx,
            duration_sec=self.spn_dur.value(),
            mode=self._get_checked_mode(),
            detail=self.sld_detail.value(),
            hand_scale=self.active_hand_scale,
            tip_x=self.active_tip_x,
            tip_y=self.active_tip_y,
            hand_id=self.active_hand_id,
            easing=self.cmb_easing.currentText(),
            enable_shadow=self.chk_shadow.isChecked(),
            enable_tilt=self.chk_tilt.isChecked(),
        )
        self.keyframes.append(new_kf)
        self.keyframes.sort(key=lambda k: k.frame_idx)
        idx = self.keyframes.index(new_kf)
        self._preview_session.clear()
        self._refresh_kf_list()
        self.list_kf.setCurrentRow(idx)
        self._refresh_timeline()
        self.refresh_frame()

    def delete_selected_kf(self):
        row = self.list_kf.currentRow()
        if 0 <= row < len(self.keyframes):
            self.keyframes.pop(row)
            self._preview_session.clear()
            self._refresh_kf_list()
            self._refresh_timeline()
            self.refresh_frame()

    def _refresh_kf_list(self):
        cur = self.list_kf.currentRow()
        self.list_kf.blockSignals(True)
        self.list_kf.clear()
        for i, kf in enumerate(self.keyframes):
            self.list_kf.addItem(QListWidgetItem(
                f"#{i + 1} · {self._fmt_time(kf.frame_idx)} · {kf.mode} ({kf.duration_sec:.1f}s)"
            ))
        if 0 <= cur < len(self.keyframes):
            self.list_kf.setCurrentRow(cur)
        self.list_kf.blockSignals(False)

    def _refresh_timeline(self):
        self.timeline.set_data(
            self.media.total_frames,
            self.media.fps,
            self.keyframes,
            self.scenes,
            selected_idx=self.list_kf.currentRow(),
            can_drag_keyframes=(self.media.source_type == "video"),
        )

    def _on_timeline_select_kf(self, idx: int):
        if 0 <= idx < len(self.keyframes):
            self.list_kf.setCurrentRow(idx)

    def _on_timeline_move_kf(self, idx: int, new_frame: int):
        if not (0 <= idx < len(self.keyframes)) or self.media.source_type != "video":
            return
        self.keyframes[idx].frame_idx = max(0, min(self.media.total_frames - 1, new_frame))
        self._preview_session.clear()
        self._refresh_kf_list()
        self._refresh_timeline()
        self.seek_to(self.keyframes[idx].frame_idx)

    def _on_list_select_kf(self, row: int):
        self.timeline.set_selected_idx(row)
        if not (0 <= row < len(self.keyframes)):
            return
        kf = self.keyframes[row]

        # Cập nhật UI inspector mà không trigger update ngược
        self._block_inspector_signals(True)
        if kf.mode in self.mode_buttons:
            self.mode_buttons[kf.mode].setChecked(True)
        self.spn_dur.setValue(kf.duration_sec)
        self.sld_detail.setValue(kf.detail)
        self.lbl_detail_val.setText(f"{kf.detail}%")
        self.cmb_easing.setCurrentText(kf.easing)
        self.chk_shadow.setChecked(kf.enable_shadow)
        self.chk_tilt.setChecked(kf.enable_tilt)

        self.active_hand_id = kf.hand_id
        self.active_tip_x = kf.tip_x
        self.active_tip_y = kf.tip_y
        self.active_hand_scale = kf.hand_scale
        self.sld_scale.setValue(int(round(kf.hand_scale * 100)))
        self.lbl_scale_val.setText(f"{int(round(kf.hand_scale * 100))}%")
        self.hand_gallery.set_selected_id(kf.hand_id)
        self._sync_hand_inspector()
        self._block_inspector_signals(False)

        # Chỉ nhảy playhead nếu playhead đang nằm ngoài vùng keyframe đó
        dur_f = max(1, int(round(kf.duration_sec * self.media.fps)))
        if not (kf.frame_idx <= self.current_frame_idx < kf.frame_idx + dur_f):
            self.seek_to(kf.frame_idx + int(round(dur_f * 0.42)))

    def _block_inspector_signals(self, block: bool):
        for w in (
            self.spn_dur, self.sld_detail, self.cmb_easing,
            self.chk_shadow, self.chk_tilt, self.sld_scale
        ):
            w.blockSignals(block)

    def _get_checked_mode(self) -> str:
        for mode, btn in self.mode_buttons.items():
            if btn.isChecked():
                return mode
        return DRAW_MODES[0]

    def _on_mode_btn_clicked(self, mode: str):
        self.mode_buttons[mode].setChecked(True)
        self._on_kf_param_changed()

    def _on_detail_slider_changed(self, val: int):
        self.lbl_detail_val.setText(f"{val}%")
        row = self.list_kf.currentRow()
        if 0 <= row < len(self.keyframes):
            self.keyframes[row].detail = val
            self._preview_session.clear()
            self._detail_debounce.start(75)

    def _on_kf_param_changed(self):
        row = self.list_kf.currentRow()
        if not (0 <= row < len(self.keyframes)):
            return
        kf = self.keyframes[row]
        new_dur = self.spn_dur.value()
        kf.mode = self._get_checked_mode()
        kf.duration_sec = new_dur
        kf.detail = self.sld_detail.value()
        kf.easing = self.cmb_easing.currentText()
        kf.enable_shadow = self.chk_shadow.isChecked()
        kf.enable_tilt = self.chk_tilt.isChecked()

        # Nếu là chuỗi ảnh, đồng bộ luôn thời lượng slide ảnh tương ứng để không bị cắt ngang nét vẽ
        if self.media.source_type == "images" and row < len(self.media.image_clips):
            starts = self.media.update_clip_duration(row, new_dur)
            for i, sf in enumerate(starts):
                if i < len(self.keyframes):
                    self.keyframes[i].frame_idx = sf
            self.scenes = list(starts)
            self._refresh_slide_list()
            self._update_project_badge()

        self._preview_session.clear()
        self._refresh_kf_list()
        self._refresh_timeline()
        self.refresh_frame()

    def _apply_effect_to_all_kfs(self):
        if not self.keyframes:
            return
        mode = self._get_checked_mode()
        dur = self.spn_dur.value()
        detail = self.sld_detail.value()
        easing = self.cmb_easing.currentText()
        sh = self.chk_shadow.isChecked()
        tilt = self.chk_tilt.isChecked()

        for i, kf in enumerate(self.keyframes):
            kf.mode = mode
            kf.duration_sec = dur
            kf.detail = detail
            kf.easing = easing
            kf.enable_shadow = sh
            kf.enable_tilt = tilt
            if self.media.source_type == "images" and i < len(self.media.image_clips):
                self.media.image_clips[i].duration_sec = dur

        if self.media.source_type == "images":
            self.media._rebuild_clip_ranges()
            starts = self.media.get_clip_starts()
            for i, sf in enumerate(starts):
                if i < len(self.keyframes):
                    self.keyframes[i].frame_idx = sf
            self.scenes = list(starts)
            self._refresh_slide_list()
            self._update_project_badge()

        self._preview_session.clear()
        self._refresh_kf_list()
        self._refresh_timeline()
        self.refresh_frame()

    # ==========================================================
    # QUẢN LÝ MẪU BÀN TAY 2D & CHỈNH ĐẦU BÚT
    # ==========================================================
    def _on_select_hand_preset(self, preset_id: str):
        preset = next((p for p in self.presets if p.id == preset_id), None)
        if preset is None:
            return
        self.active_hand_id = preset.id
        self.active_tip_x = preset.tip_x
        self.active_tip_y = preset.tip_y
        self.active_hand_scale = preset.default_scale

        self._block_inspector_signals(True)
        self.sld_scale.setValue(int(round(preset.default_scale * 100)))
        self.lbl_scale_val.setText(f"{int(round(preset.default_scale * 100))}%")
        self._block_inspector_signals(False)

        self._sync_hand_inspector()
        self._apply_hand_to_keyframes()

    def _on_scale_slider_changed(self, val: int):
        self.lbl_scale_val.setText(f"{val}%")
        self.active_hand_scale = val / 100.0
        self._apply_hand_to_keyframes()

    def _on_tip_picked(self, tx: int, ty: int):
        self.active_tip_x = tx
        self.active_tip_y = ty
        self.lbl_tip_coords.setText(f"Tọa độ đầu bút: ({tx}, {ty})")
        self._apply_hand_to_keyframes()

    def _auto_detect_current_tip(self):
        bgra = self.hand_map.get(self.active_hand_id)
        if bgra is None:
            return
        tx, ty = detect_pen_tip(bgra)
        self.active_tip_x, self.active_tip_y = tx, ty
        self._sync_hand_inspector()
        self._apply_hand_to_keyframes()

    def _sync_hand_inspector(self):
        bgra = self.hand_map.get(self.active_hand_id)
        self.tip_picker.set_hand(bgra, self.active_tip_x, self.active_tip_y)
        self.lbl_tip_coords.setText(f"Tọa độ đầu bút: ({self.active_tip_x}, {self.active_tip_y})")

    def _apply_hand_to_keyframes(self):
        if self.chk_hand_apply_all.isChecked():
            for kf in self.keyframes:
                kf.hand_id = self.active_hand_id
                kf.tip_x = self.active_tip_x
                kf.tip_y = self.active_tip_y
                kf.hand_scale = self.active_hand_scale
        else:
            row = self.list_kf.currentRow()
            if 0 <= row < len(self.keyframes):
                kf = self.keyframes[row]
                kf.hand_id = self.active_hand_id
                kf.tip_x = self.active_tip_x
                kf.tip_y = self.active_tip_y
                kf.hand_scale = self.active_hand_scale
        self.refresh_frame()

    def load_custom_hand(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Chọn ảnh bàn tay riêng (PNG trong suốt hoặc ảnh nền phẳng)",
            "", "Ảnh (*.png *.jpg *.jpeg *.webp)"
        )
        if not path:
            return
        raw = imread_unicode(path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            QMessageBox.warning(self, "Lỗi", "Không đọc được file ảnh này.")
            return

        # Tự tách nền nếu ảnh chưa có kênh trong suốt
        bgra = remove_solid_background(raw, tolerance=36.0, feather=16.0)
        bgra = trim_and_pad_hand(bgra, target_max_dim=560, pad=18)
        tx, ty = detect_pen_tip(bgra)

        custom_id = f"custom_{len(self.presets) + 1}"
        name = os.path.splitext(os.path.basename(path))[0][:18]
        preset = HandPreset(
            id=custom_id,
            name=name,
            category="Tùy chỉnh",
            filename="",
            tip_x=tx,
            tip_y=ty,
            default_scale=0.65,
            description=f"Mẫu tay tải lên: {os.path.basename(path)}",
        )
        self.presets.append(preset)
        self.hand_map[custom_id] = bgra
        self.hand_gallery.set_presets(self.presets, self.hand_map, custom_id)
        self._on_select_hand_preset(custom_id)

    # ==========================================================
    # XUẤT VIDEO MP4
    # ==========================================================
    def export_video(self):
        if self.media.source_type == "none":
            QMessageBox.information(self, "Thông báo", "Chưa có Video hoặc Ảnh để xuất.")
            return
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Lưu Video Whiteboard", "whiteboard_output.mp4", "MP4 Video (*.mp4)"
        )
        if not out_path:
            return
        if self.is_playing:
            self.toggle_play()

        self.tabs.setCurrentIndex(2)
        self.btn_export.setEnabled(False)
        self.btn_top_export.setEnabled(False)
        self.btn_cancel_export.setEnabled(True)
        self.progress.setValue(0)

        crf = int(self.cmb_quality.currentData() or 20)
        default_hand = self.hand_map.get(self.active_hand_id)
        self.exporter = ExportThread(
            self.media,
            out_path,
            self.keyframes,
            self.hand_map,
            default_hand,
            self.chk_audio.isChecked(),
            quality_crf=crf,
        )
        self.exporter.progress.connect(self.progress.setValue)
        self.exporter.status_text.connect(self.lbl_export_status.setText)
        self.exporter.done.connect(self._on_export_done)
        self.exporter.start()

    def _cancel_export(self):
        if self.exporter is not None:
            self.exporter.cancel()
            self.lbl_export_status.setText("Đang hủy...")

    def _on_export_done(self, ok: bool, msg: str):
        self.btn_export.setEnabled(True)
        self.btn_top_export.setEnabled(True)
        self.btn_cancel_export.setEnabled(False)
        self.lbl_export_status.setText("Hoàn tất!" if ok else "Đã dừng.")
        if ok:
            QMessageBox.information(self, "Xuất Video Thành Công", msg)
        else:
            QMessageBox.warning(self, "Thông báo", msg)
