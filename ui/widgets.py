import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal, QPointF, QRectF, QSize
from PyQt6.QtGui import (
    QColor, QDragEnterEvent, QDropEvent, QFont, QImage,
    QPainter, QPen, QPixmap, QPolygonF
)
from PyQt6.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QVBoxLayout, QWidget
)

from core.engine import Keyframe
from core.hand_tools import HandPreset, build_checkerboard


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def np_bgr_to_pixmap(bgr: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


def composite_on_checker(bgra: np.ndarray, size: Tuple[int, int]) -> QPixmap:
    tw, th = size
    ih, iw = bgra.shape[:2]
    scale = min((tw - 10) / max(1, iw), (th - 10) / max(1, ih))
    nw, nh = max(8, int(round(iw * scale))), max(8, int(round(ih * scale)))
    rs = cv2.resize(bgra, (nw, nh), interpolation=cv2.INTER_AREA)
    chk = build_checkerboard(tw, th, cell=9)
    ox, oy = (tw - nw) // 2, (th - nh) // 2
    a = rs[:, :, 3:4].astype(np.float32) / 255.0
    chk[oy:oy + nh, ox:ox + nw] = np.clip(
        rs[:, :, :3].astype(np.float32) * a + chk[oy:oy + nh, ox:ox + nw].astype(np.float32) * (1.0 - a),
        0, 255
    ).astype(np.uint8)
    return np_bgr_to_pixmap(chk)


class PreviewStageWidget(QFrame):
    """Khung xem trước chính hỗ trợ kéo-thả Video/Ảnh và hiển thị màn hình chào gọn đẹp."""
    files_dropped = pyqtSignal(list)
    request_load_video = pyqtSignal()
    request_load_images = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setObjectName("StageBox")
        self.setAcceptDrops(True)
        self.setMinimumSize(580, 380)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        # Trạng thái trống (onboarding)
        self.empty_box = QWidget()
        e_lay = QVBoxLayout(self.empty_box)
        e_lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        e_lay.setSpacing(12)

        lbl_icon = QLabel("✍️")
        lbl_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_icon.setStyleSheet("font-size: 44px; background: transparent;")

        lbl_title = QLabel("Tạo Video Vẽ Tay Whiteboard Mượt & Chân Thực")
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_title.setStyleSheet("font-size: 17px; font-weight: 700; color: #F1F5F9; background: transparent;")

        lbl_sub = QLabel(
            "Kéo thả Video hoặc Ảnh vào đây để bắt đầu.\n"
            "Mỗi ảnh sẽ tự động được phân tích nét vẽ & gắn Keyframe chuẩn."
        )
        lbl_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_sub.setStyleSheet("color: #94A3B8; font-size: 13px; background: transparent;")

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        btn_row.setAlignment(Qt.AlignmentFlag.AlignCenter)

        b_vid = QPushButton("🎬 Nhập Video")
        b_vid.setObjectName("primary")
        b_vid.setMinimumWidth(140)
        b_vid.clicked.connect(self.request_load_video.emit)

        b_img = QPushButton("🖼 Ghép Ảnh Thành Video")
        b_img.setObjectName("accent")
        b_img.setMinimumWidth(180)
        b_img.clicked.connect(self.request_load_images.emit)

        btn_row.addWidget(b_vid)
        btn_row.addWidget(b_img)

        e_lay.addWidget(lbl_icon)
        e_lay.addWidget(lbl_title)
        e_lay.addWidget(lbl_sub)
        e_lay.addLayout(btn_row)

        # Khung ảnh xem trước
        self.lbl_frame = QLabel()
        self.lbl_frame.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.lbl_frame.setStyleSheet("background: transparent;")
        self.lbl_frame.hide()

        lay.addWidget(self.empty_box)
        lay.addWidget(self.lbl_frame)
        self._last_pixmap: Optional[QPixmap] = None

    def show_frame(self, bgr: np.ndarray):
        self.empty_box.hide()
        self.lbl_frame.show()
        self._last_pixmap = np_bgr_to_pixmap(bgr)
        self._apply_scaled_pixmap()

    def show_empty(self):
        self._last_pixmap = None
        self.lbl_frame.clear()
        self.lbl_frame.hide()
        self.empty_box.show()

    def _apply_scaled_pixmap(self):
        if self._last_pixmap is None or self.lbl_frame.isHidden():
            return
        sz = self.lbl_frame.size()
        if sz.width() > 10 and sz.height() > 10:
            scaled = self._last_pixmap.scaled(
                sz, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
            self.lbl_frame.setPixmap(scaled)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._apply_scaled_pixmap()

    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e: QDropEvent):
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        valid = [
            p for p in paths
            if os.path.splitext(p)[1].lower() in (VIDEO_EXTS | IMAGE_EXTS)
        ]
        if valid:
            self.files_dropped.emit(valid)


class TimelineWidget(QWidget):
    """Timeline trực quan có thước thời gian, chọn & kéo Keyframe, và tua playhead mượt."""
    seek_requested = pyqtSignal(int)
    keyframe_selected = pyqtSignal(int)
    keyframe_moved = pyqtSignal(int, int)  # (kf_index, new_frame_idx)

    MODE_COLORS = {
        "Vẽ Nét + Tô Màu": QColor("#F59E0B"),
        "Chỉ Vẽ Nét": QColor("#10B981"),
        "Phác Thảo Thực Tế": QColor("#A855F7"),
        "Live Sketch": QColor("#A855F7"),
        "Cọ Tô Nhanh": QColor("#06B6D4"),
    }

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(92)
        self.setMaximumHeight(104)
        self.setMouseTracking(True)

        self.total_frames = 100
        self.fps = 30.0
        self.current_frame = 0
        self.keyframes: List[Keyframe] = []
        self.scenes: List[int] = []
        self.selected_kf_idx: int = -1
        self.can_drag_keyframes: bool = False

        self._drag_mode: Optional[str] = None  # "scrub" | "move_kf"
        self._drag_kf_idx: int = -1
        self._margin_x = 16

    def set_data(
        self,
        total_frames: int,
        fps: float,
        keyframes: List[Keyframe],
        scenes: List[int],
        selected_idx: int = -1,
        can_drag_keyframes: bool = False
    ):
        self.total_frames = max(1, total_frames)
        self.fps = max(1.0, fps)
        self.keyframes = keyframes
        self.scenes = scenes
        self.selected_kf_idx = selected_idx
        self.can_drag_keyframes = can_drag_keyframes
        self.update()

    def set_selected_idx(self, idx: int):
        self.selected_kf_idx = idx
        self.update()

    def set_current_frame(self, frame_idx: int):
        self.current_frame = max(0, min(self.total_frames - 1, frame_idx))
        self.update()

    def _x_to_frame(self, x: float) -> int:
        usable = max(1, self.width() - 2 * self._margin_x)
        ratio = max(0.0, min(1.0, (x - self._margin_x) / usable))
        return int(round(ratio * max(0, self.total_frames - 1)))

    def _frame_to_x(self, f_idx: int) -> float:
        usable = max(1, self.width() - 2 * self._margin_x)
        return self._margin_x + (f_idx / max(1, self.total_frames - 1)) * usable

    def _hit_test_keyframe(self, x: float, y: float) -> int:
        if y < 24:
            return -1
        for i, kf in enumerate(self.keyframes):
            x1 = self._frame_to_x(kf.frame_idx)
            dur_f = int(round(kf.duration_sec * self.fps))
            x2 = max(x1 + 12, self._frame_to_x(min(self.total_frames - 1, kf.frame_idx + dur_f)))
            if (x1 - 6) <= x <= x2 and 24 <= y <= self.height() - 8:
                return i
        return -1

    def mousePressEvent(self, e):
        if e.button() != Qt.MouseButton.LeftButton:
            return
        x, y = e.position().x(), e.position().y()
        hit = self._hit_test_keyframe(x, y)
        if hit >= 0 and y >= 32:
            self.selected_kf_idx = hit
            self.keyframe_selected.emit(hit)
            if self.can_drag_keyframes and abs(x - self._frame_to_x(self.keyframes[hit].frame_idx)) <= 12:
                self._drag_mode = "move_kf"
                self._drag_kf_idx = hit
            else:
                self._drag_mode = "scrub"
                self.seek_requested.emit(self._x_to_frame(x))
            self.update()
        else:
            self._drag_mode = "scrub"
            self.seek_requested.emit(self._x_to_frame(x))

    def mouseMoveEvent(self, e):
        x, y = e.position().x(), e.position().y()
        if self._drag_mode == "scrub":
            self.seek_requested.emit(self._x_to_frame(x))
        elif self._drag_mode == "move_kf" and 0 <= self._drag_kf_idx < len(self.keyframes):
            nf = self._x_to_frame(x)
            self.keyframe_moved.emit(self._drag_kf_idx, nf)
        else:
            hit = self._hit_test_keyframe(x, y)
            if hit >= 0:
                kf = self.keyframes[hit]
                sec = kf.frame_idx / self.fps
                self.setToolTip(
                    f"Key #{hit + 1} · {int(sec // 60):02d}:{sec % 60:04.1f}\n"
                    f"Chế độ: {kf.mode} ({kf.duration_sec:.1f}s)"
                )
            else:
                self.setToolTip("Nhấp hoặc kéo để tua khung hình")

    def mouseReleaseEvent(self, e):
        self._drag_mode = None
        self._drag_kf_idx = -1

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        p.fillRect(0, 0, w, h, QColor("#12151E"))
        p.setPen(QPen(QColor("#232838"), 1))
        p.drawLine(0, 0, w, 0)

        # Thước thời gian ở phần trên (y: 0..24)
        p.fillRect(0, 0, w, 24, QColor("#171B26"))
        total_sec = self.total_frames / self.fps
        if total_sec > 0:
            step_sec = 1.0
            for cand in [1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0]:
                if total_sec / cand <= 10:
                    step_sec = cand
                    break
            p.setFont(QFont("Consolas", 9))
            t = 0.0
            while t <= total_sec + 0.01:
                tx = self._frame_to_x(int(round(t * self.fps)))
                p.setPen(QPen(QColor("#394258"), 1))
                p.drawLine(QPointF(tx, 16), QPointF(tx, 24))
                p.setPen(QColor("#7C879E"))
                label = f"{int(t // 60):02d}:{int(t % 60):02d}"
                p.drawText(QRectF(tx - 18, 2, 40, 14), Qt.AlignmentFlag.AlignCenter, label)
                t += step_sec

        # Đường track nền
        track_y, track_h = 34, h - 46
        p.setBrush(QColor("#1B202E"))
        p.setPen(QPen(QColor("#2A3144"), 1))
        p.drawRoundedRect(QRectF(self._margin_x, track_y, w - 2 * self._margin_x, track_h), 6, 6)

        # Vạch cảnh
        p.setPen(QPen(QColor("#38BDF8"), 1.3, Qt.PenStyle.DashLine))
        for s in self.scenes:
            sx = self._frame_to_x(s)
            p.drawLine(QPointF(sx, track_y - 4), QPointF(sx, track_y + track_h + 4))

        # Khối Keyframe
        p.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        for i, kf in enumerate(self.keyframes):
            x1 = self._frame_to_x(kf.frame_idx)
            dur_f = int(round(kf.duration_sec * self.fps))
            x2 = self._frame_to_x(min(self.total_frames - 1, kf.frame_idx + dur_f))
            bw = max(14.0, x2 - x1)

            base_col = self.MODE_COLORS.get(kf.mode, QColor("#F59E0B"))
            fill_col = QColor(base_col)
            fill_col.setAlpha(190 if i == self.selected_kf_idx else 135)

            p.setBrush(fill_col)
            if i == self.selected_kf_idx:
                p.setPen(QPen(QColor("#FFFFFF"), 2.0))
            else:
                p.setPen(QPen(base_col.lighter(120), 1.0))
            p.drawRoundedRect(QRectF(x1, track_y + 3, bw, track_h - 6), 5, 5)

            if bw > 28:
                p.setPen(QColor("#0B0D12") if kf.mode != "Phác Thảo Thực Tế" else QColor("#FFFFFF"))
                p.drawText(
                    QRectF(x1 + 6, track_y + 3, bw - 10, track_h - 6),
                    Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                    f"#{i + 1}"
                )

            # Diamond đầu keyframe
            diamond = QPolygonF([
                QPointF(x1, track_y - 4), QPointF(x1 + 6, track_y + 2),
                QPointF(x1, track_y + 8), QPointF(x1 - 6, track_y + 2)
            ])
            p.setBrush(QColor("#FBBF24") if i != self.selected_kf_idx else QColor("#FFFFFF"))
            p.setPen(QPen(QColor("#0F1117"), 1.2))
            p.drawPolygon(diamond)

        # Playhead
        px = self._frame_to_x(self.current_frame)
        p.setPen(QPen(QColor("#F43F5E"), 2.2))
        p.drawLine(QPointF(px, 12), QPointF(px, h - 4))
        head = QPolygonF([
            QPointF(px - 6, 12), QPointF(px + 6, 12),
            QPointF(px + 6, 20), QPointF(px, 26), QPointF(px - 6, 20)
        ])
        p.setBrush(QColor("#F43F5E"))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPolygon(head)


class HandCardWidget(QFrame):
    """Thẻ chọn mẫu bàn tay 2D tách nền kèm ảnh xem trước trên nền ca-rô."""
    clicked = pyqtSignal(str)

    def __init__(self, preset: HandPreset, bgra: np.ndarray):
        super().__init__()
        self.preset_id = preset.id
        self.setObjectName("HandCard")
        self.setProperty("selected", "false")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(preset.description or preset.name)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)

        lbl_img = QLabel()
        lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_img.setFixedHeight(86)
        lbl_img.setPixmap(composite_on_checker(bgra, (132, 82)))
        lbl_img.setStyleSheet("border-radius: 5px; background: transparent;")

        lbl_name = QLabel(preset.name)
        lbl_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_name.setStyleSheet("font-size: 11px; font-weight: 600; color: #E2E8F0; background: transparent;")

        lbl_cat = QLabel(preset.category)
        lbl_cat.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_cat.setStyleSheet("font-size: 10px; color: #94A3B8; background: transparent;")

        lay.addWidget(lbl_img)
        lay.addWidget(lbl_name)
        lay.addWidget(lbl_cat)

    def set_selected(self, selected: bool):
        self.setProperty("selected", "true" if selected else "false")
        self.style().unpolish(self)
        self.style().polish(self)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.preset_id)


class HandGalleryWidget(QWidget):
    """Lưới chọn các mẫu bàn tay 2D tách nền."""
    preset_selected = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(8)
        self.cards: Dict[str, HandCardWidget] = {}

    def set_presets(self, presets: List[HandPreset], hand_map: Dict[str, np.ndarray], selected_id: str):
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.cards.clear()

        for i, p in enumerate(presets):
            bgra = hand_map.get(p.id)
            if bgra is None:
                continue
            card = HandCardWidget(p, bgra)
            card.set_selected(p.id == selected_id)
            card.clicked.connect(self._on_card_clicked)
            self.cards[p.id] = card
            self.grid.addWidget(card, i // 2, i % 2)

    def _on_card_clicked(self, preset_id: str):
        self.set_selected_id(preset_id)
        self.preset_selected.emit(preset_id)

    def set_selected_id(self, preset_id: str):
        for pid, card in self.cards.items():
            card.set_selected(pid == preset_id)


class HandTipPickerWidget(QLabel):
    """Khung chỉnh tọa độ đầu bút trực quan: click trực tiếp lên ảnh bàn tay để đặt tâm đầu bút."""
    tip_changed = pyqtSignal(int, int)

    def __init__(self):
        super().__init__()
        self.setFixedSize(296, 175)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setStyleSheet("border: 1px solid #2D3548; border-radius: 6px; background: #141822;")
        self.setToolTip("Nhấp hoặc kéo chuột để đặt đúng vị trí đầu bút trên ảnh bàn tay")

        self._bgra: Optional[np.ndarray] = None
        self._tip_x = 20
        self._tip_y = 20
        self._scale = 1.0
        self._ox = 0
        self._oy = 0
        self._dragging = False

    def set_hand(self, bgra: Optional[np.ndarray], tip_x: int, tip_y: int):
        self._bgra = bgra
        self._tip_x = tip_x
        self._tip_y = tip_y
        self._redraw()

    def set_tip(self, tip_x: int, tip_y: int):
        self._tip_x = tip_x
        self._tip_y = tip_y
        self._redraw()

    def _redraw(self):
        w, h = self.width(), self.height()
        chk = build_checkerboard(w, h, cell=10)
        if self._bgra is not None:
            ih, iw = self._bgra.shape[:2]
            self._scale = min((w - 16) / max(1, iw), (h - 16) / max(1, ih))
            nw, nh = max(8, int(round(iw * self._scale))), max(8, int(round(ih * self._scale)))
            rs = cv2.resize(self._bgra, (nw, nh), interpolation=cv2.INTER_AREA)
            self._ox = (w - nw) // 2
            self._oy = (h - nh) // 2
            a = rs[:, :, 3:4].astype(np.float32) / 255.0
            chk[self._oy:self._oy + nh, self._ox:self._ox + nw] = np.clip(
                rs[:, :, :3].astype(np.float32) * a
                + chk[self._oy:self._oy + nh, self._ox:self._ox + nw].astype(np.float32) * (1.0 - a),
                0, 255
            ).astype(np.uint8)

            cx = int(round(self._ox + self._tip_x * self._scale))
            cy = int(round(self._oy + self._tip_y * self._scale))
            cv2.circle(chk, (cx, cy), 10, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.drawMarker(chk, (cx, cy), (60, 76, 255), cv2.MARKER_CROSS, 18, 2)

        self.setPixmap(np_bgr_to_pixmap(chk))

    def _pick_at(self, px: float, py: float):
        if self._bgra is None or self._scale <= 0:
            return
        ih, iw = self._bgra.shape[:2]
        tx = int(np.clip(round((px - self._ox) / self._scale), 0, iw - 1))
        ty = int(np.clip(round((py - self._oy) / self._scale), 0, ih - 1))
        self._tip_x, self._tip_y = tx, ty
        self._redraw()
        self.tip_changed.emit(tx, ty)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._pick_at(e.position().x(), e.position().y())

    def mouseMoveEvent(self, e):
        if self._dragging:
            self._pick_at(e.position().x(), e.position().y())

    def mouseReleaseEvent(self, e):
        self._dragging = False
