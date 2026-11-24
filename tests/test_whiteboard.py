import os
import sys
import tempfile
import unittest

import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.engine import DRAW_MODES, Keyframe, WhiteboardEngine
from core.hand_tools import (
    detect_pen_tip, imread_unicode, imwrite_unicode,
    load_hand_presets, remove_solid_background
)
from core.media import MediaProvider
from core.workers import ExportThread
from ui.main_window import WhiteboardStudioWindow


def make_test_illustration(w: int = 800, h: int = 480) -> np.ndarray:
    """Tạo ảnh minh họa nhiều màu & nét rõ để kiểm thử engine."""
    img = np.full((h, w, 3), (250, 250, 252), dtype=np.uint8)
    # Mặt trời vàng
    cv2.circle(img, (650, 100), 48, (40, 200, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (650, 100), 48, (30, 80, 140), 3, cv2.LINE_AA)
    # Ngôi nhà mái đỏ tường kem
    cv2.rectangle(img, (180, 210), (430, 390), (195, 225, 245), -1)
    cv2.rectangle(img, (180, 210), (430, 390), (30, 35, 45), 4, cv2.LINE_AA)
    roof = np.array([[155, 210], [305, 110], [455, 210]], dtype=np.int32)
    cv2.fillPoly(img, [roof], (70, 75, 220))
    cv2.polylines(img, [roof], True, (30, 35, 45), 4, cv2.LINE_AA)
    # Cửa gỗ
    cv2.rectangle(img, (275, 290), (335, 390), (50, 100, 160), -1)
    cv2.rectangle(img, (275, 290), (335, 390), (25, 30, 40), 3, cv2.LINE_AA)
    # Cây xanh
    cv2.rectangle(img, (545, 270), (575, 390), (45, 85, 130), -1)
    cv2.circle(img, (560, 225), 62, (65, 175, 70), -1, cv2.LINE_AA)
    cv2.circle(img, (560, 225), 62, (30, 60, 30), 3, cv2.LINE_AA)
    # Chữ
    cv2.putText(img, "STUDIO 2D", (175, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (40, 45, 65), 4, cv2.LINE_AA)
    return img


class TestWhiteboardSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)
        cls.assets_dir = os.path.join(ROOT_DIR, "assets")
        cls.hands_dir = os.path.join(cls.assets_dir, "hands")

    def test_presets_exist_and_transparent(self):
        presets = load_hand_presets(self.hands_dir)
        self.assertGreaterEqual(len(presets), 8)
        for p in presets:
            path = os.path.join(self.hands_dir, p.filename)
            self.assertTrue(os.path.exists(path), f"Missing {path}")
            bgra = imread_unicode(path, cv2.IMREAD_UNCHANGED)
            self.assertIsNotNone(bgra)
            self.assertEqual(bgra.shape[2], 4)
            # Kiểm tra góc ảnh phải trong suốt
            self.assertLess(int(bgra[0, 0, 3]), 15)
            # Kiểm tra tọa độ đầu bút nằm trong khung hình
            self.assertTrue(0 <= p.tip_x < bgra.shape[1])
            self.assertTrue(0 <= p.tip_y < bgra.shape[0])

    def test_chroma_key_and_tip_detector(self):
        # Ảnh test nền xanh lá với hình vẽ màu đỏ
        synth = np.full((240, 240, 3), (0, 255, 0), dtype=np.uint8)
        cv2.line(synth, (30, 50), (210, 210), (30, 40, 220), 18, cv2.LINE_AA)
        bgra = remove_solid_background(synth)
        self.assertLess(int(bgra[5, 5, 3]), 10)
        self.assertGreater(int(bgra[130, 120, 3]), 230)
        tx, ty = detect_pen_tip(bgra)
        self.assertLess(tx, 50)
        self.assertLess(abs(ty - 50), 25)

    def test_engine_all_modes_and_incremental_consistency(self):
        img = make_test_illustration()
        presets = load_hand_presets(self.hands_dir)
        hand = imread_unicode(os.path.join(self.hands_dir, presets[0].filename), cv2.IMREAD_UNCHANGED)

        plan = WhiteboardEngine.prepare_sketch_plan(img, detail=65, paper_style="white_studio")
        self.assertGreater(len(plan["line_pts"]), 50)
        self.assertGreater(len(plan["color_pts"]), 20)

        for mode in DRAW_MODES:
            kf = Keyframe(
                frame_idx=0, duration_sec=4.0, mode=mode, detail=65,
                hand_scale=0.6, tip_x=presets[0].tip_x, tip_y=presets[0].tip_y, hand_id=presets[0].id
            )
            sess = {}
            out_prev = None
            for p in [0.02, 0.35, 0.70, 0.98]:
                out = WhiteboardEngine.render_frame(plan, p, hand, kf, session_state=sess)
                self.assertEqual(out.shape, img.shape)
                out_prev = out

            # Kết thúc tại 0.98 cho chế độ có tô màu phải khớp gần như 100% ảnh gốc
            if mode in ("Vẽ Nét + Tô Màu", "Phác Thảo Thực Tế", "Cọ Tô Nhanh"):
                diff = float(np.mean(np.abs(out_prev.astype(np.float32) - img.astype(np.float32))))
                self.assertLess(diff, 2.0, f"Mode {mode} final diff too high: {diff}")

        # Kiểm tra ảnh trắng trơn không gây lỗi
        blank = np.full((320, 480, 3), 255, dtype=np.uint8)
        plan_blank = WhiteboardEngine.prepare_sketch_plan(blank, detail=50)
        kf_b = Keyframe(frame_idx=0, mode="Vẽ Nét + Tô Màu")
        out_b = WhiteboardEngine.render_frame(plan_blank, 0.5, hand, kf_b)
        self.assertEqual(out_b.shape, blank.shape)

    def test_media_and_export_mp4(self):
        with tempfile.TemporaryDirectory() as tmp:
            p1 = os.path.join(tmp, "s1.png")
            p2 = os.path.join(tmp, "s2.png")
            imwrite_unicode(p1, make_test_illustration())
            imwrite_unicode(p2, cv2.cvtColor(make_test_illustration(), cv2.COLOR_BGR2RGB))

            media = MediaProvider()
            media.set_paper_and_resolution("warm_sketch", "hd_16_9")
            ok, added = media.add_images([p1, p2], sec_each=1.2)
            self.assertTrue(ok)
            self.assertEqual(len(added), 2)
            # Kiểm tra đồng bộ thời lượng slide
            starts = media.update_clip_duration(0, 1.5)
            self.assertEqual(starts[0], 0)
            self.assertEqual(starts[1], int(round(1.5 * media.fps)))

            presets = load_hand_presets(self.hands_dir)
            hand = imread_unicode(os.path.join(self.hands_dir, presets[0].filename), cv2.IMREAD_UNCHANGED)
            hand_map = {presets[0].id: hand}
            kfs = [
                Keyframe(frame_idx=starts[0], duration_sec=1.5, mode="Vẽ Nét + Tô Màu",
                         tip_x=presets[0].tip_x, tip_y=presets[0].tip_y, hand_id=presets[0].id),
                Keyframe(frame_idx=starts[1], duration_sec=1.2, mode="Chỉ Vẽ Nét",
                         tip_x=presets[0].tip_x, tip_y=presets[0].tip_y, hand_id=presets[0].id),
            ]
            out_mp4 = os.path.join(tmp, "out.mp4")
            exp = ExportThread(media, out_mp4, kfs, hand_map, hand, keep_audio=False, quality_crf=23)
            res = {}
            exp.done.connect(lambda ok_f, msg: res.update({"ok": ok_f, "msg": msg}))
            exp.run()  # Chạy đồng bộ trong test
            self.assertTrue(res.get("ok"), res.get("msg"))
            self.assertTrue(os.path.exists(out_mp4))
            self.assertGreater(os.path.getsize(out_mp4), 10000)

            cap = cv2.VideoCapture(out_mp4)
            self.assertTrue(cap.isOpened())
            self.assertGreaterEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 60)
            cap.release()

    def test_ui_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            p1 = os.path.join(tmp, "slide_tiếng_việt.png")
            imwrite_unicode(p1, make_test_illustration())

            win = WhiteboardStudioWindow(assets_dir=self.assets_dir)
            win.add_image_files([p1])
            self.assertEqual(len(win.keyframes), 1)

            # Đổi mẫu tay
            win._on_select_hand_preset("pencil_classic")
            self.assertEqual(win.keyframes[0].hand_id, "pencil_classic")

            # Đổi chế độ & thông số
            win._on_mode_btn_clicked("Chỉ Vẽ Nét")
            self.assertEqual(win.keyframes[0].mode, "Chỉ Vẽ Nét")

            win.spn_dur.setValue(5.0)
            self.assertEqual(win.keyframes[0].duration_sec, 5.0)

            # Xóa keyframe
            win.delete_selected_kf()
            self.assertEqual(len(win.keyframes), 0)
            win.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
