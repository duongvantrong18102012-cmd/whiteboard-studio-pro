from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import threading

import cv2
import numpy as np

from core.hand_tools import imread_unicode


PAPER_STYLES: Dict[str, Tuple[str, Tuple[int, int, int], bool]] = {
    # id: (label, BGR base color, is_dark_board)
    "white_studio": ("Giấy Trắng Studio", (248, 249, 251), False),
    "warm_sketch": ("Giấy Mỹ Thuật Kem", (236, 244, 249), False),
    "kraft_paper": ("Giấy Kraft Cổ Điển", (174, 203, 224), False),
    "chalkboard": ("Bảng Phấn Xanh", (46, 56, 36), True),
}

CANVAS_RESOLUTIONS: Dict[str, Tuple[str, int, int]] = {
    "hd_16_9": ("1280×720 (16:9 HD)", 1280, 720),
    "fhd_16_9": ("1920×1080 (Full HD)", 1920, 1080),
    "shorts_9_16": ("1080×1920 (9:16 Dọc)", 1080, 1920),
    "square_1_1": ("1080×1080 (1:1 Vuông)", 1080, 1080),
}


@dataclass
class ImageClip:
    path: str
    raw_bgr: np.ndarray
    duration_sec: float = 4.5
    fitted_bgr: Optional[np.ndarray] = None


class PaperGenerator:
    """Sinh và lưu cache nền giấy vẽ có vân giấy nhẹ tự nhiên."""
    _cache: Dict[Tuple[int, int, str], np.ndarray] = {}

    @classmethod
    def get_paper(cls, width: int, height: int, style: str = "white_studio") -> np.ndarray:
        key = (width, height, style)
        if key in cls._cache:
            return cls._cache[key]

        _, bgr, is_dark = PAPER_STYLES.get(style, PAPER_STYLES["white_studio"])
        base = np.full((height, width, 3), bgr, dtype=np.int16)

        # Tạo vân giấy tần số thấp + hạt mịn bằng RNG cố định để không nhấp nháy
        rng = np.random.default_rng(42)
        small_w, small_h = max(64, width // 4), max(64, height // 4)
        grain = rng.normal(0.0, 2.8 if not is_dark else 3.6, (small_h, small_w, 1)).astype(np.float32)
        grain = cv2.resize(grain, (width, height), interpolation=cv2.INTER_LINEAR)

        # Vignette nhẹ cực êm ở 4 góc tạo chiều sâu studio
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        nx = (xx / max(1, width - 1)) * 2.0 - 1.0
        ny = (yy / max(1, height - 1)) * 2.0 - 1.0
        r2 = (nx * nx + ny * ny) * 0.5
        vig = np.clip(r2 * (6.0 if not is_dark else 12.0), 0.0, 16.0)

        paper = np.clip(base - vig[:, :, None] + grain[:, :, None], 0, 255).astype(np.uint8)
        if len(cls._cache) > 8:
            cls._cache.clear()
        cls._cache[key] = paper
        return paper


class MediaProvider:
    """Quản lý nguồn Video và Chuỗi Ảnh với bộ đệm LRU và đọc tuần tự tối ưu."""

    def __init__(self):
        self.source_type: str = "none"  # "none" | "video" | "images"
        self.video_path: Optional[str] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self._cap_lock = threading.Lock()
        self._last_decoded_idx: int = -1
        self._frame_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_limit: int = 36

        self.fps: float = 30.0
        self.width: int = 1280
        self.height: int = 720
        self.total_frames: int = 0

        self.paper_style: str = "white_studio"
        self.resolution_key: str = "hd_16_9"
        self.image_clips: List[ImageClip] = []
        self._clip_ranges: List[Tuple[int, int]] = []  # [(start_frame, end_frame_excl), ...]

    def close(self):
        with self._cap_lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self._last_decoded_idx = -1
            self._frame_cache.clear()

    def load_video(self, path: str) -> bool:
        self.close()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return False
        self.source_type = "video"
        self.video_path = path
        self.cap = cap
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.fps = fps if 5.0 <= fps <= 120.0 else 30.0
        self.width = max(64, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        self.height = max(64, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.total_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.image_clips.clear()
        self._clip_ranges.clear()
        return True

    def _fit_to_canvas(self, img: np.ndarray, tw: int, th: int) -> np.ndarray:
        """Căn ảnh vừa khung giấy vẽ với lề thoáng đẹp, giữ nguyên tỷ lệ."""
        ih, iw = img.shape[:2]
        pad_ratio = 0.94  # chừa viền 3% mỗi phía để tay vẽ thoáng đẹp
        scale = min((tw * pad_ratio) / max(1, iw), (th * pad_ratio) / max(1, ih))
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        resized = cv2.resize(img, (nw, nh), interpolation=interp)

        canvas = PaperGenerator.get_paper(tw, th, self.paper_style).copy()
        ox = (tw - nw) // 2
        oy = (th - nh) // 2

        # Nếu ảnh đầu vào có nền gần trắng và giấy không phải bảng tối, hòa trộn mềm viền
        _, _, is_dark = PAPER_STYLES.get(self.paper_style, PAPER_STYLES["white_studio"])
        if not is_dark and self.paper_style != "white_studio":
            # Hòa trộn multiply nhẹ với nền giấy mỹ thuật / kraft
            roi = canvas[oy:oy + nh, ox:ox + nw].astype(np.float32) / 255.0
            fg = resized.astype(np.float32) / 255.0
            canvas[oy:oy + nh, ox:ox + nw] = np.clip(roi * fg * 255.0, 0, 255).astype(np.uint8)
        else:
            canvas[oy:oy + nh, ox:ox + nw] = resized
        return canvas

    def set_paper_and_resolution(self, paper_style: Optional[str] = None, resolution_key: Optional[str] = None):
        if paper_style and paper_style in PAPER_STYLES:
            self.paper_style = paper_style
        if resolution_key and resolution_key in CANVAS_RESOLUTIONS and self.source_type != "video":
            self.resolution_key = resolution_key
            _, self.width, self.height = CANVAS_RESOLUTIONS[resolution_key]

        if self.source_type == "images":
            for clip in self.image_clips:
                clip.fitted_bgr = self._fit_to_canvas(clip.raw_bgr, self.width, self.height)
            self._rebuild_clip_ranges()

    def _rebuild_clip_ranges(self):
        self._clip_ranges.clear()
        cursor = 0
        for clip in self.image_clips:
            n_frames = max(15, int(round(clip.duration_sec * self.fps)))
            self._clip_ranges.append((cursor, cursor + n_frames))
            cursor += n_frames
        self.total_frames = max(1, cursor)

    def add_images(self, paths: List[str], sec_each: float = 4.5) -> Tuple[bool, List[Tuple[int, float]]]:
        """Thêm một hoặc nhiều ảnh vào dự án; trả về danh sách (start_frame, duration_sec) mới thêm."""
        if self.source_type != "images":
            self.close()
            self.source_type = "images"
            self.video_path = None
            self.fps = 30.0
            _, self.width, self.height = CANVAS_RESOLUTIONS.get(self.resolution_key, ("HD", 1280, 720))
            self.image_clips.clear()
            self._clip_ranges.clear()

        added: List[Tuple[int, float]] = []
        cursor = self._clip_ranges[-1][1] if self._clip_ranges else 0

        for p in paths:
            img = imread_unicode(p, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            # Nếu ảnh PNG có kênh alpha, dán lên màu nền giấy hiện tại
            if len(img.shape) == 3 and img.shape[2] == 4:
                _, paper_bgr, _ = PAPER_STYLES.get(self.paper_style, PAPER_STYLES["white_studio"])
                bg = np.full((img.shape[0], img.shape[1], 3), paper_bgr, dtype=np.float32)
                a = img[:, :, 3:4].astype(np.float32) / 255.0
                img = np.clip(img[:, :, :3].astype(np.float32) * a + bg * (1.0 - a), 0, 255).astype(np.uint8)
            elif len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            fitted = self._fit_to_canvas(img, self.width, self.height)
            clip = ImageClip(path=p, raw_bgr=img, duration_sec=sec_each, fitted_bgr=fitted)
            self.image_clips.append(clip)
            n_frames = max(15, int(round(sec_each * self.fps)))
            self._clip_ranges.append((cursor, cursor + n_frames))
            added.append((cursor, sec_each))
            cursor += n_frames

        self.total_frames = max(1, cursor) if self.image_clips else 0
        return len(added) > 0, added

    def update_clip_duration(self, clip_index: int, new_duration_sec: float) -> List[int]:
        """Cập nhật thời lượng slide ảnh và trả về danh sách start_frame mới của tất cả slide."""
        if 0 <= clip_index < len(self.image_clips):
            self.image_clips[clip_index].duration_sec = max(1.0, float(new_duration_sec))
            self._rebuild_clip_ranges()
        return [r[0] for r in self._clip_ranges]

    def get_clip_starts(self) -> List[int]:
        return [r[0] for r in self._clip_ranges]

    def read_frame(self, idx: int) -> Optional[np.ndarray]:
        if self.source_type == "images":
            if not self.image_clips:
                return None
            idx = max(0, min(self.total_frames - 1, idx))
            for i, (s, e) in enumerate(self._clip_ranges):
                if s <= idx < e:
                    return self.image_clips[i].fitted_bgr.copy()
            return self.image_clips[-1].fitted_bgr.copy()

        if self.source_type == "video" and self.cap is not None:
            idx = max(0, min(self.total_frames - 1, idx))
            if idx in self._frame_cache:
                self._frame_cache.move_to_end(idx)
                return self._frame_cache[idx].copy()

            with self._cap_lock:
                if idx == self._last_decoded_idx + 1:
                    ret, frame = self.cap.read()
                elif 0 < (idx - self._last_decoded_idx) <= 4:
                    ret, frame = False, None
                    for _ in range(idx - self._last_decoded_idx):
                        ret, frame = self.cap.read()
                        if not ret:
                            break
                else:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = self.cap.read()

                if not ret or frame is None:
                    return None
                self._last_decoded_idx = idx
                self._frame_cache[idx] = frame
                if len(self._frame_cache) > self._cache_limit:
                    self._frame_cache.popitem(last=False)
                return frame.copy()

        return None
