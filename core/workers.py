import os
import shutil
import subprocess
from typing import Dict, List, Optional

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from core.engine import Keyframe, WhiteboardEngine
from core.media import MediaProvider


class SceneAnalyzerThread(QThread):
    progress = pyqtSignal(int)
    finished_scenes = pyqtSignal(list)

    def __init__(self, media: MediaProvider):
        super().__init__()
        self.media = media
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        if self.media.source_type == "images":
            scenes = self.media.get_clip_starts()
            self.progress.emit(100)
            self.finished_scenes.emit(scenes)
            return

        if not self.media.video_path:
            self.finished_scenes.emit([])
            return

        cap = cv2.VideoCapture(self.media.video_path)
        total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        min_gap = int(fps * 1.8)

        scenes = [0]
        prev_hist = None
        step = max(1, int(round(fps / 6.0)))
        idx = 0

        while idx < total and not self._cancelled:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            hsv = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            if prev_hist is not None:
                diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                if diff > 0.40 and (idx - scenes[-1]) >= min_gap:
                    scenes.append(idx)
            prev_hist = hist
            self.progress.emit(min(100, int(idx / total * 100)))
            idx += step

        cap.release()
        self.progress.emit(100)
        if not self._cancelled:
            self.finished_scenes.emit(scenes)


class ExportThread(QThread):
    progress = pyqtSignal(int)
    status_text = pyqtSignal(str)
    done = pyqtSignal(bool, str)

    def __init__(
        self,
        media: MediaProvider,
        output_path: str,
        keyframes: List[Keyframe],
        hand_map: Dict[str, np.ndarray],
        default_hand_bgra: Optional[np.ndarray],
        keep_audio: bool,
        quality_crf: int = 19,
        out_fps: Optional[float] = None
    ):
        super().__init__()
        self.media = media
        self.output_path = output_path
        self.keyframes = sorted(keyframes, key=lambda k: k.frame_idx)
        self.hand_map = hand_map
        self.default_hand_bgra = default_hand_bgra
        self.keep_audio = keep_audio and (media.source_type == "video")
        self.quality_crf = quality_crf
        self.out_fps = out_fps or media.fps
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        fps = self.media.fps
        w, h = self.media.width, self.media.height
        total = max(1, self.media.total_frames)

        temp_out = self.output_path + ".temp_render.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(temp_out, fourcc, fps, (w, h))
        if not writer.isOpened():
            self.done.emit(False, "Không thể khởi tạo bộ ghi video MP4.")
            return

        cap_v = cv2.VideoCapture(self.media.video_path) if self.media.source_type == "video" else None
        try:
            def get_frame(i: int) -> Optional[np.ndarray]:
                if cap_v is not None:
                    cap_v.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, f = cap_v.read()
                    return f if ret else None
                return self.media.read_frame(i)

            self.status_text.emit("Đang phân tích nét vẽ & mảng màu cho các Keyframe...")
            plans: Dict[int, dict] = {}
            for ki, kf in enumerate(self.keyframes):
                if self._cancelled:
                    break
                f = get_frame(kf.frame_idx)
                if f is not None:
                    plans[kf.frame_idx] = WhiteboardEngine.prepare_sketch_plan(
                        f, kf.detail, self.media.paper_style
                    )
                self.progress.emit(min(12, int((ki + 1) / max(1, len(self.keyframes)) * 12)))

            if cap_v is not None:
                cap_v.set(cv2.CAP_PROP_POS_FRAMES, 0)

            session_state: dict = {}
            self.status_text.emit("Đang render khung hình vẽ tay...")

            for idx in range(total):
                if self._cancelled:
                    break

                if cap_v is not None:
                    ret, frame = cap_v.read()
                    if not ret or frame is None:
                        break
                else:
                    frame = self.media.read_frame(idx)
                    if frame is None:
                        break

                active_kf: Optional[Keyframe] = None
                kf_prog = 0.0
                for kf in self.keyframes:
                    dur_f = max(1, int(round(kf.duration_sec * fps)))
                    if kf.frame_idx <= idx < kf.frame_idx + dur_f:
                        active_kf = kf
                        kf_prog = (idx - kf.frame_idx) / max(1, dur_f - 1)
                        break

                if active_kf is not None and active_kf.frame_idx in plans:
                    hand_bgra = self.hand_map.get(active_kf.hand_id, self.default_hand_bgra)
                    frame = WhiteboardEngine.render_frame(
                        plans[active_kf.frame_idx],
                        kf_prog,
                        hand_bgra,
                        active_kf,
                        session_state=session_state,
                    )

                writer.write(frame)
                if idx % 4 == 0 or idx == total - 1:
                    pct = 12 + int((idx + 1) / total * 78)
                    self.progress.emit(min(90, pct))
        finally:
            if cap_v is not None:
                cap_v.release()
            writer.release()

        if self._cancelled:
            if os.path.exists(temp_out):
                os.remove(temp_out)
            self.done.emit(False, "Đã hủy xuất video.")
            return

        ffmpeg_exe = shutil.which("ffmpeg")
        if ffmpeg_exe:
            self.status_text.emit("Đang nén H.264 chất lượng cao bằng FFmpeg...")
            try:
                cmd = [ffmpeg_exe, "-y", "-i", temp_out]
                if self.keep_audio and self.media.video_path:
                    cmd.extend(["-i", self.media.video_path])
                cmd.extend([
                    "-c:v", "libx264",
                    "-preset", "medium",
                    "-crf", str(self.quality_crf),
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                ])
                if self.keep_audio and self.media.video_path:
                    cmd.extend(["-c:a", "aac", "-b:a", "192k", "-map", "0:v:0", "-map", "1:a:0?", "-shortest"])
                cmd.append(self.output_path)
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                if os.path.exists(temp_out):
                    os.remove(temp_out)
                msg = f"Đã xuất video MP4 (H.264) thành công:\n{self.output_path}"
            except Exception:
                os.replace(temp_out, self.output_path)
                msg = f"Đã xuất video MP4 thành công (fallback mp4v):\n{self.output_path}"
        else:
            os.replace(temp_out, self.output_path)
            msg = f"Đã xuất video MP4 thành công:\n{self.output_path}"

        self.progress.emit(100)
        self.status_text.emit("Hoàn tất!")
        self.done.emit(True, msg)
