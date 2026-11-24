from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.media import PaperGenerator, PAPER_STYLES


DRAW_MODES = [
    "Vẽ Nét + Tô Màu",
    "Chỉ Vẽ Nét",
    "Phác Thảo Thực Tế",
    "Cọ Tô Nhanh",
]


@dataclass
class Keyframe:
    frame_idx: int
    duration_sec: float = 4.5
    mode: str = "Vẽ Nét + Tô Màu"
    detail: int = 60
    hand_scale: float = 0.65
    tip_x: int = 20
    tip_y: int = 20
    hand_id: str = "marker_vector"
    easing: str = "Tự nhiên (Ease)"  # "Tự nhiên (Ease)" | "Đều (Linear)"
    enable_shadow: bool = True
    enable_tilt: bool = True


def _ease_in_out_sine(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 0.5 - 0.5 * float(np.cos(np.pi * t))


def _ease_out_cubic(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1.0 - (1.0 - t) ** 3


def _ease_in_cubic(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t ** 3


def _chaikin_smooth(pts: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Làm mượt đường polyline bằng thuật toán cắt góc Chaikin."""
    cur = pts.astype(np.float32)
    for _ in range(iterations):
        if len(cur) < 3:
            break
        out = [cur[0]]
        for i in range(len(cur) - 1):
            p0, p1 = cur[i], cur[i + 1]
            out.append(0.75 * p0 + 0.25 * p1)
            out.append(0.25 * p0 + 0.75 * p1)
        out.append(cur[-1])
        cur = np.asarray(out, dtype=np.float32)
    return cur


def _resample_arc_length(pts: np.ndarray, step_px: float = 4.0) -> np.ndarray:
    """Lấy mẫu lại đường nét theo khoảng cách cung đều nhau để tốc độ vẽ tay mượt và ổn định."""
    if len(pts) < 2:
        return pts.astype(np.float32)
    diffs = np.diff(pts, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    cum = np.concatenate(([0.0], np.cumsum(seg_lens)))
    total = float(cum[-1])
    if total < 1.0:
        return pts[:1].astype(np.float32)
    n_samples = max(2, int(np.ceil(total / step_px)) + 1)
    sample_dists = np.linspace(0.0, total, n_samples)
    xs = np.interp(sample_dists, cum, pts[:, 0])
    ys = np.interp(sample_dists, cum, pts[:, 1])
    return np.stack([xs, ys], axis=1).astype(np.float32)


def _zhang_suen_thin(img_u8: np.ndarray, max_iters: int = 8) -> np.ndarray:
    """Làm mảnh đường nét về khung xương 1-pixel bằng Zhang-Suen vectorized NumPy."""
    bw = (img_u8 > 0).astype(np.uint8)
    p = np.pad(bw, 1, mode="constant")
    for _ in range(max_iters):
        changed = False
        for step in (0, 1):
            p2 = p[:-2, 1:-1]
            p3 = p[:-2, 2:]
            p4 = p[1:-1, 2:]
            p5 = p[2:, 2:]
            p6 = p[2:, 1:-1]
            p7 = p[2:, :-2]
            p8 = p[1:-1, :-2]
            p9 = p[:-2, :-2]
            c = p[1:-1, 1:-1]
            n = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            t = (
                ((p2 == 0) & (p3 == 1)).astype(np.uint8)
                + ((p3 == 0) & (p4 == 1)).astype(np.uint8)
                + ((p4 == 0) & (p5 == 1)).astype(np.uint8)
                + ((p5 == 0) & (p6 == 1)).astype(np.uint8)
                + ((p6 == 0) & (p7 == 1)).astype(np.uint8)
                + ((p7 == 0) & (p8 == 1)).astype(np.uint8)
                + ((p8 == 0) & (p9 == 1)).astype(np.uint8)
                + ((p9 == 0) & (p2 == 1)).astype(np.uint8)
            )
            if step == 0:
                cond = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                cond = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            m = (c == 1) & (n >= 2) & (n <= 6) & (t == 1) & cond
            if np.any(m):
                p[1:-1, 1:-1][m] = 0
                changed = True
        if not changed:
            break
    return (p[1:-1, 1:-1] * 255).astype(np.uint8)


def _trace_skeleton_paths(skel: np.ndarray, min_len: int = 8) -> List[np.ndarray]:
    """
    Lần theo khung xương 1-pixel (8-connected) để mỗi nét chỉ được vẽ ĐÚNG 1 LẦN
    (tránh lỗi findContours đi khứ hồi 2 lần trên đường 1px), ưu tiên hướng đi thẳng qua giao điểm.
    """
    h, w = skel.shape
    ys, xs = np.where(skel > 0)
    if len(xs) == 0:
        return []

    visited = np.zeros((h, w), dtype=np.uint8)
    # Tính bậc (số láng giềng 8-connected) cho từng pixel xương
    bin_mask = (skel > 0).astype(np.uint8)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    deg_map = cv2.filter2D(bin_mask, cv2.CV_8U, kernel, borderType=cv2.BORDER_CONSTANT)

    # Sắp xếp điểm bắt đầu: điểm đầu mút (deg==1) trước, sau đó trên xuống dưới
    endpoints = [(int(y), int(x)) for y, x in zip(ys, xs) if deg_map[y, x] == 1]
    others = [(int(y), int(x)) for y, x in zip(ys, xs) if deg_map[y, x] != 1]
    seeds = endpoints + others

    neighbors_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    strokes: List[np.ndarray] = []

    for sy, sx in seeds:
        if visited[sy, sx] or bin_mask[sy, sx] == 0:
            continue

        path = [(sx, sy)]
        visited[sy, sx] = 1
        cy, cx = sy, sx
        prev_dy, prev_dx = 0.0, 0.0

        while True:
            best_pt = None
            best_score = -1e9
            for dy, dx in neighbors_8:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and bin_mask[ny, nx] and not visited[ny, nx]:
                    # Ưu tiên giữ hướng tiếp tuyến mượt + ưu tiên bước thẳng 4-connected nhẹ
                    step_len = 1.414 if (dy != 0 and dx != 0) else 1.0
                    if prev_dx == 0.0 and prev_dy == 0.0:
                        score = -step_len
                    else:
                        dot = (dx * prev_dx + dy * prev_dy) / step_len
                        score = dot * 3.0 - step_len * 0.25
                    if score > best_score:
                        best_score = score
                        best_pt = (ny, nx, dy, dx)

            if best_pt is None:
                break

            ny, nx, dy, dx = best_pt
            visited[ny, nx] = 1
            path.append((nx, ny))
            prev_dx = 0.6 * prev_dx + 0.4 * dx
            prev_dy = 0.6 * prev_dy + 0.4 * dy
            cy, cx = ny, nx

        if len(path) >= min_len:
            strokes.append(np.asarray(path, dtype=np.float32))

    return strokes


def _chain_strokes_naturally(
    raw_strokes: List[np.ndarray],
    w: int,
    h: int
) -> List[np.ndarray]:
    """
    Sắp xếp nét tự nhiên như họa sĩ thực thụ:
    - Vẽ các nét khung chính (dài & rõ) trước để định hình bức tranh
    - Sau đó nối nét chi tiết theo lân cận gần nhất, có ưu tiên nhẹ từ trên-trái xuống dưới-phải.
    """
    if not raw_strokes:
        return []

    lengths = [float(np.sum(np.hypot(np.diff(s[:, 0]), np.diff(s[:, 1])))) for s in raw_strokes]
    cutoff = float(np.percentile(lengths, 65)) if len(lengths) > 6 else 0.0

    major = [s for s, ln in zip(raw_strokes, lengths) if ln >= cutoff]
    minor = [s for s, ln in zip(raw_strokes, lengths) if ln < cutoff]

    def chain_group(group: List[np.ndarray], start_pt: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        if not group:
            return [], start_pt
        rem = list(group)
        ordered: List[np.ndarray] = []
        cur_pt = start_pt
        # Nếu bắt đầu mới, ưu tiên nét gần vùng trên-trái nhất
        if len(ordered) == 0 and np.allclose(start_pt, [0.0, 0.0]):
            cur_pt = np.array([w * 0.15, h * 0.15], dtype=np.float32)

        while rem:
            best_i = 0
            best_rev = False
            best_cost = 1e18
            for i, s in enumerate(rem):
                p0, p1 = s[0], s[-1]
                d0 = float((p0[0] - cur_pt[0]) ** 2 + (p0[1] - cur_pt[1]) ** 2)
                d1 = float((p1[0] - cur_pt[0]) ** 2 + (p1[1] - cur_pt[1]) ** 2)
                # Thiên vị nhẹ cho nét ở phía trên để dòng vẽ tự nhiên
                bias0 = p0[1] * 12.0
                bias1 = p1[1] * 12.0
                if d0 + bias0 < best_cost:
                    best_cost, best_i, best_rev = d0 + bias0, i, False
                if d1 + bias1 < best_cost:
                    best_cost, best_i, best_rev = d1 + bias1, i, True
            chosen = rem.pop(best_i)
            if best_rev:
                chosen = chosen[::-1].copy()
            ordered.append(chosen)
            cur_pt = chosen[-1]
        return ordered, cur_pt

    ord_major, last_pt = chain_group(major, np.array([0.0, 0.0], dtype=np.float32))
    ord_minor, _ = chain_group(minor, last_pt)
    return ord_major + ord_minor


def _build_timeline_events(
    strokes: List[np.ndarray],
    weights: Optional[List[float]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Chuyển danh sách nét thành chuỗi bước liên tục (N, 2) kèm cờ `is_draw` (True=vẽ, False=nhấc bút di chuyển)
    và `stroke_w` độ dày nét tại mỗi bước. Loại bỏ hoàn toàn hiện tượng tay nhảy cóc tức thời!
    """
    if not strokes:
        empty_pts = np.zeros((1, 2), dtype=np.float32)
        return empty_pts, np.zeros(1, dtype=bool), np.ones(1, dtype=np.float32) * 2.0

    pts_list: List[np.ndarray] = []
    draw_flags: List[np.ndarray] = []
    width_vals: List[np.ndarray] = []

    prev_end: Optional[np.ndarray] = None
    for idx, stroke in enumerate(strokes):
        w_val = weights[idx] if weights and idx < len(weights) else 2.0
        if prev_end is not None:
            gap = float(np.hypot(stroke[0, 0] - prev_end[0], stroke[0, 1] - prev_end[1]))
            if gap > 22.0:
                n_lift = int(np.clip(np.sqrt(gap) * 0.72, 3, 14))
                ts = np.linspace(0.0, 1.0, n_lift + 2)[1:-1]
                # Ease-in-out cubic + độ võng nhẹ khi nhấc bút
                ease_ts = np.array([_ease_in_out_sine(float(t)) for t in ts], dtype=np.float32)
                arc_y = np.sin(ts * np.pi) * min(28.0, gap * 0.08)
                lift_pts = prev_end[None, :] + (stroke[0] - prev_end)[None, :] * ease_ts[:, None]
                lift_pts[:, 1] -= arc_y.astype(np.float32)
                pts_list.append(lift_pts)
                draw_flags.append(np.zeros(len(lift_pts), dtype=bool))
                width_vals.append(np.zeros(len(lift_pts), dtype=np.float32))

        pts_list.append(stroke)
        flags = np.ones(len(stroke), dtype=bool)
        flags[0] = False  # Điểm đầu của nét mới đặt bút xuống (không nối từ điểm nhấc bút)
        draw_flags.append(flags)
        width_vals.append(np.full(len(stroke), w_val, dtype=np.float32))
        prev_end = stroke[-1]

    all_pts = np.concatenate(pts_list, axis=0)
    all_draw = np.concatenate(draw_flags, axis=0)
    all_w = np.concatenate(width_vals, axis=0)
    return all_pts, all_draw, all_w


class HandCompositor:
    """Bộ ghép bàn tay 2D có khử răng cưa, bóng đổ mềm trên giấy và bộ đệm xoay lượng tử hóa siêu nhanh."""
    _cache: Dict[Tuple[int, float], Tuple[np.ndarray, np.ndarray]] = {}
    _rot_cache: Dict[Tuple[int, float, float], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    @classmethod
    def _get_scaled_hand_and_shadow(
        cls,
        fg_bgra: np.ndarray,
        scale: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        q_scale = round(float(scale), 3)
        key = (id(fg_bgra), q_scale)
        if key in cls._cache:
            return cls._cache[key]

        h_fg, w_fg = fg_bgra.shape[:2]
        nw = max(16, int(round(w_fg * q_scale)))
        nh = max(16, int(round(h_fg * q_scale)))
        interp = cv2.INTER_AREA if q_scale < 1.0 else cv2.INTER_LINEAR
        scaled = cv2.resize(fg_bgra, (nw, nh), interpolation=interp)

        # Tạo bóng đổ mềm từ kênh alpha của bàn tay
        a = scaled[:, :, 3].astype(np.float32) / 255.0
        k_blur = max(7, int(min(nw, nh) * 0.08) | 1)
        shadow = cv2.GaussianBlur(a, (k_blur, k_blur), 0)

        if len(cls._cache) > 16:
            cls._cache.clear()
        cls._cache[key] = (scaled, shadow)
        return scaled, shadow

    @classmethod
    def _get_rotated_assets(
        cls,
        fg_bgra: np.ndarray,
        scale: float,
        tip_x: int,
        tip_y: int,
        tilt_deg: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        q_scale = round(float(scale), 3)
        q_tilt = round(float(tilt_deg) * 2.0) / 2.0  # lượng tử hóa mỗi 0.5 độ để tái sử dụng cache
        rkey = (id(fg_bgra), q_scale, q_tilt)
        if rkey in cls._rot_cache:
            return cls._rot_cache[rkey]

        scaled, shadow = cls._get_scaled_hand_and_shadow(fg_bgra, q_scale)
        nh, nw = scaled.shape[:2]
        if abs(q_tilt) > 0.2:
            M = cv2.getRotationMatrix2D((tip_x * q_scale, tip_y * q_scale), q_tilt, 1.0)
            hand_rot = cv2.warpAffine(
                scaled, M, (nw, nh),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0),
            )
            shadow_rot = cv2.warpAffine(
                shadow, M, (nw, nh),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )
        else:
            hand_rot = scaled
            shadow_rot = shadow

        fg_bgr = np.ascontiguousarray(hand_rot[:, :, :3])
        w_fg = np.ascontiguousarray(hand_rot[:, :, 3].astype(np.float32) * (1.0 / 255.0))
        w_bg = np.ascontiguousarray(1.0 - w_fg)
        sh_f = np.ascontiguousarray(shadow_rot.astype(np.float32))

        if len(cls._rot_cache) > 64:
            cls._rot_cache.clear()
        cls._rot_cache[rkey] = (fg_bgr, w_fg, w_bg, sh_f)
        return fg_bgr, w_fg, w_bg, sh_f

    @classmethod
    def composite(
        cls,
        bg_bgr: np.ndarray,
        fg_bgra: Optional[np.ndarray],
        pos_x: float,
        pos_y: float,
        tip_x: int,
        tip_y: int,
        scale: float,
        tilt_deg: float = 0.0,
        lift_amount: float = 0.0,
        enable_shadow: bool = True
    ) -> np.ndarray:
        if fg_bgra is None:
            return bg_bgr

        fg_bgr, w_fg, w_bg_arr, sh_f = cls._get_rotated_assets(
            fg_bgra, scale, tip_x, tip_y, tilt_deg
        )
        nh, nw = fg_bgr.shape[:2]
        tx = tip_x * scale
        ty = tip_y * scale

        out = bg_bgr.copy()
        h_bg, w_bg = out.shape[:2]

        # 1. Vẽ bóng đổ mềm trên giấy trước bằng SIMD blendLinear với vùng đen
        if enable_shadow:
            sh_dx = int(round(10 + lift_amount * 14))
            sh_dy = int(round(14 + lift_amount * 18))
            sh_opacity = max(0.08, 0.20 - lift_amount * 0.07)
            cls._blend_shadow(
                out, sh_f,
                int(round(pos_x - tx)) + sh_dx,
                int(round(pos_y - ty)) + sh_dy,
                sh_opacity,
                w_bg, h_bg
            )

        # 2. Vẽ bàn tay 2D sắc nét bằng cv2.blendLinear trên vùng ROI nhỏ
        lift_dy = -int(round(lift_amount * 5.0))
        x1 = int(round(pos_x - tx))
        y1 = int(round(pos_y - ty)) + lift_dy
        x2, y2 = x1 + nw, y1 + nh

        if x2 <= 0 or y2 <= 0 or x1 >= w_bg or y1 >= h_bg:
            return out

        bx1, by1 = max(0, x1), max(0, y1)
        bx2, by2 = min(w_bg, x2), min(h_bg, y2)
        fx1, fy1 = bx1 - x1, by1 - y1
        fx2, fy2 = fx1 + (bx2 - bx1), fy1 + (by2 - by1)

        roi_fg = fg_bgr[fy1:fy2, fx1:fx2]
        roi_bg = out[by1:by2, bx1:bx2]
        wf = np.ascontiguousarray(w_fg[fy1:fy2, fx1:fx2])
        wb = np.ascontiguousarray(w_bg_arr[fy1:fy2, fx1:fx2])
        out[by1:by2, bx1:bx2] = cv2.blendLinear(roi_fg, roi_bg, wf, wb)
        return out

    @staticmethod
    def _blend_shadow(
        target: np.ndarray,
        shadow_alpha: np.ndarray,
        x1: int,
        y1: int,
        opacity: float,
        w_bg: int,
        h_bg: int
    ):
        nh, nw = shadow_alpha.shape[:2]
        x2, y2 = x1 + nw, y1 + nh
        if x2 <= 0 or y2 <= 0 or x1 >= w_bg or y1 >= h_bg:
            return
        bx1, by1 = max(0, x1), max(0, y1)
        bx2, by2 = min(w_bg, x2), min(h_bg, y2)
        fx1, fy1 = bx1 - x1, by1 - y1
        fx2, fy2 = fx1 + (bx2 - bx1), fy1 + (by2 - by1)

        sh = np.ascontiguousarray(shadow_alpha[fy1:fy2, fx1:fx2] * opacity)
        inv_sh = np.ascontiguousarray(1.0 - sh)
        roi = target[by1:by2, bx1:bx2]
        zeros = np.zeros_like(roi)
        target[by1:by2, bx1:bx2] = cv2.blendLinear(zeros, roi, sh, inv_sh)


class WhiteboardEngine:
    """Engine vẽ tay tự nhiên với khung xương liên kết mượt, tô màu theo hướng khối và bộ đệm tăng tốc."""

    @staticmethod
    def _build_hatching_strokes(
        smooth_gray: np.ndarray,
        scale_x: float,
        scale_y: float,
        is_dark_board: bool
    ) -> List[np.ndarray]:
        """Tạo nét đánh bóng chéo (cross-hatching) cho vùng tối đậm trong chế độ Chỉ Vẽ Nét."""
        ah, aw = smooth_gray.shape
        dark_mask = (smooth_gray < 82).astype(np.uint8)
        if is_dark_board:
            # Trên bảng đen phấn trắng, đánh bóng nhẹ ở vùng sáng trung bình
            dark_mask = ((smooth_gray > 145) & (smooth_gray < 235)).astype(np.uint8)

        # Loại bỏ các nét mảnh nhỏ (đã được vẽ bởi outline) để chỉ đánh bóng mảng tối lớn
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        core_dark = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, k)
        num_cc, cc_map, stats, _ = cv2.connectedComponentsWithStats(core_dark, connectivity=8)

        hatches: List[np.ndarray] = []
        spacing = 7
        for cid in range(1, num_cc):
            if stats[cid, cv2.CC_STAT_AREA] < 140:
                continue
            x0 = int(stats[cid, cv2.CC_STAT_LEFT])
            y0 = int(stats[cid, cv2.CC_STAT_TOP])
            cw = int(stats[cid, cv2.CC_STAT_WIDTH])
            ch = int(stats[cid, cv2.CC_STAT_HEIGHT])
            comp = (cc_map == cid)

            pts: List[List[float]] = []
            flip = False
            for d in range(x0 + y0, x0 + cw + y0 + ch + 1, spacing):
                seg = []
                xs = max(x0, d - (y0 + ch - 1))
                xe = min(x0 + cw - 1, d - y0)
                for px in range(xs, xe + 1, 2):
                    py = d - px
                    if 0 <= py < ah and comp[py, px]:
                        seg.append([px * scale_x, py * scale_y])
                if len(seg) >= 3:
                    if flip:
                        seg.reverse()
                    # Ngắt thành đoạn riêng biệt nếu quá xa để không kéo dây giữa các vùng lồi lõm
                    hatches.append(_resample_arc_length(np.asarray(seg, dtype=np.float32), step_px=5.0))
                    flip = not flip
        return hatches

    @staticmethod
    def _build_color_strokes(
        frame_bgr: np.ndarray,
        is_dark_board: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Phân vùng màu theo không gian Lab (300px), xác định trục hướng chính (PCA) của từng mảng
        và sinh đường cọ lướt hữu cơ mượt mà theo đúng dáng hình.
        """
        h, w = frame_bgr.shape[:2]
        dw = 260
        dh = max(80, int(round(dw * h / max(1, w))))
        small = cv2.resize(frame_bgr, (dw, dh), interpolation=cv2.INTER_AREA)
        small = cv2.bilateralFilter(small, 5, 32, 32)
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2Lab)

        Z = lab.reshape((-1, 3)).astype(np.float32)
        k_clusters = 8
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.2)
        _, labels, centers = cv2.kmeans(Z, k_clusters, None, criteria, 1, cv2.KMEANS_PP_CENTERS)
        labels = labels.reshape((dh, dw))

        brush_w = max(24, int(round(min(w, h) * 0.058)))
        sx, sy = w / dw, h / dh
        step_scan = max(4, int(round(brush_w * 0.50 / max(1.0, sx))))

        cluster_order = sorted(range(k_clusters), key=lambda c: int(np.sum(labels == c)), reverse=True)
        raw_color_strokes: List[np.ndarray] = []

        for c in cluster_order:
            L_val, a_val, b_val = centers[c]
            # Bỏ qua màu giấy trắng sáng hoặc màu bảng tối trùng với nền
            if not is_dark_board and L_val > 236 and abs(a_val - 128) < 10 and abs(b_val - 128) < 10:
                continue
            if is_dark_board and L_val < 52 and abs(a_val - 128) < 12 and abs(b_val - 128) < 12:
                continue

            mask_c = (labels == c).astype(np.uint8)
            k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_c = cv2.morphologyEx(mask_c, cv2.MORPH_CLOSE, k_close)
            num_cc, cc_map, stats, _ = cv2.connectedComponentsWithStats(mask_c, connectivity=8)

            comps = []
            for cid in range(1, num_cc):
                if stats[cid, cv2.CC_STAT_AREA] >= 22:
                    comps.append((stats[cid, cv2.CC_STAT_LEFT], stats[cid, cv2.CC_STAT_TOP], cid))
            comps.sort()

            for _, _, cid in comps:
                ys_c, xs_c = np.where(cc_map == cid)
                if len(xs_c) < 12:
                    continue

                # Tìm hướng trục chính (PCA) của mảng màu để tô xuôi chiều vật thể
                pts_c = np.stack([xs_c.astype(np.float32), ys_c.astype(np.float32)], axis=1)
                mean_pt = pts_c.mean(axis=0)
                centered = pts_c - mean_pt
                cov = np.cov(centered.T)
                eigvals, eigvecs = np.linalg.eigh(cov)
                # Vector chính (hướng dài nhất của mảng màu)
                major_axis = eigvecs[:, int(np.argmax(eigvals))]
                angle = float(np.arctan2(major_axis[1], major_axis[0]))

                # Xoay tọa độ vùng màu theo trục chính để quét zích-zắc song song
                cos_a, sin_a = np.cos(-angle), np.sin(-angle)
                u = centered[:, 0] * cos_a - centered[:, 1] * sin_a
                v = centered[:, 0] * sin_a + centered[:, 1] * cos_a

                v_min, v_max = float(v.min()), float(v.max())
                u_min, u_max = float(u.min()), float(u.max())

                # Lưới quét theo các dải v song song với trục chính
                comp_mask = (cc_map == cid)
                stroke_pts: List[List[float]] = []
                flip = False

                v_cur = v_min
                while v_cur <= v_max + 0.5:
                    band = np.abs(v - v_cur) <= max(1.6, step_scan * 0.45)
                    if np.any(band):
                        u_band = u[band]
                        u_lo, u_hi = float(u_band.min()), float(u_band.max())
                        if u_hi - u_lo >= 1.5:
                            n_seg = max(3, int((u_hi - u_lo) / 3.0) + 1)
                            us = np.linspace(u_lo, u_hi, n_seg)
                            if flip:
                                us = us[::-1]
                            for ui in us:
                                # Thêm gợn cọ hữu cơ nhẹ
                                v_wobble = v_cur + np.sin(ui * 0.25) * 1.1
                                gx = mean_pt[0] + ui * np.cos(angle) - v_wobble * np.sin(angle)
                                gy = mean_pt[1] + ui * np.sin(angle) + v_wobble * np.cos(angle)
                                ix = int(np.clip(round(gx), 0, dw - 1))
                                iy = int(np.clip(round(gy), 0, dh - 1))
                                if comp_mask[iy, ix] or len(stroke_pts) == 0:
                                    stroke_pts.append([
                                        float(np.clip(gx * sx, 0, w - 1)),
                                        float(np.clip(gy * sy, 0, h - 1)),
                                    ])
                            flip = not flip
                    v_cur += step_scan

                if len(stroke_pts) >= 3:
                    smoothed = _chaikin_smooth(np.asarray(stroke_pts, dtype=np.float32), iterations=1)
                    raw_color_strokes.append(_resample_arc_length(smoothed, step_px=6.5))

        ordered = _chain_strokes_naturally(raw_color_strokes, w, h)
        c_pts, c_draw, c_w = _build_timeline_events(ordered)
        return c_pts, c_draw, c_w, brush_w

    @classmethod
    def prepare_sketch_plan(
        cls,
        frame_bgr: np.ndarray,
        detail: int = 60,
        paper_style: str = "white_studio"
    ) -> dict:
        h, w = frame_bgr.shape[:2]
        aw = min(w, 840)
        scale_x = w / max(1, aw)
        ah = max(60, int(round(h / scale_x)))
        scale_y = h / max(1, ah)

        small = cv2.resize(frame_bgr, (aw, ah), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        smooth = cv2.bilateralFilter(gray, 5, 34, 34)

        # Kết hợp Canny + đường mực mảnh để nét chữ và đường viền rõ ràng không bị rỗng đôi
        t_high = max(32, 222 - int(detail * 1.7))
        t_low = max(12, int(t_high * 0.40))
        edges = cv2.Canny(smooth, t_low, t_high)

        _, _, is_dark_board = PAPER_STYLES.get(paper_style, PAPER_STYLES["white_studio"])
        if not is_dark_board:
            dark_bin = cv2.adaptiveThreshold(
                smooth, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 11
            )
            dark_bin = cv2.medianBlur(dark_bin, 3)
            dist_ink = cv2.distanceTransform(dark_bin, cv2.DIST_L2, 3)
            # Chỉ lấy các nét mảnh/chữ viết thực sự (tránh tạo khung xương nhện trong mảng đen đặc)
            thin_ink = ((dark_bin > 0) & (dist_ink <= 4.2)).astype(np.uint8) * 255
            combined = cv2.bitwise_or(edges, thin_ink)
        else:
            combined = edges

        # Thu gọn về khung xương 1-pixel bằng Zhang-Suen thinning
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
            skel = cv2.ximgproc.thinning(combined, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        else:
            skel = _zhang_suen_thin(combined, max_iters=8)

        min_len = max(5, int(14 - detail * 0.08))
        raw_strokes = _trace_skeleton_paths(skel, min_len=min_len)

        scaled_strokes: List[np.ndarray] = []
        weights: List[float] = []
        for s in raw_strokes:
            s_full = s.copy()
            s_full[:, 0] *= scale_x
            s_full[:, 1] *= scale_y
            smoothed = _chaikin_smooth(s_full, iterations=2)
            resampled = _resample_arc_length(smoothed, step_px=4.0)
            if len(resampled) >= 2:
                # Ước lượng độ đậm nét từ độ dài & tương phản cục bộ
                w_stroke = 2.4 if len(resampled) > 35 else 1.8
                scaled_strokes.append(resampled)
                weights.append(w_stroke)

        ordered_strokes = _chain_strokes_naturally(scaled_strokes, w, h)
        # Tính lại weights theo thứ tự đã chain
        len_to_w = {id(s): wt for s, wt in zip(scaled_strokes, weights)}
        ordered_weights = [len_to_w.get(id(s), 2.0) for s in ordered_strokes]

        line_pts, line_draw, line_w = _build_timeline_events(ordered_strokes, ordered_weights)

        # Chuẩn bị nét đánh bóng (hatching) riêng cho chế độ Chỉ Vẽ Nét
        hatches = cls._build_hatching_strokes(smooth, scale_x, scale_y, is_dark_board)
        hatch_ordered = _chain_strokes_naturally(hatches, w, h)
        hatch_pts, hatch_draw, hatch_w = _build_timeline_events(
            hatch_ordered, [1.5] * len(hatch_ordered)
        )

        # Chuẩn bị chuỗi tô màu theo vùng Lab + PCA
        c_pts, c_draw, c_w, brush_w = cls._build_color_strokes(frame_bgr, is_dark_board)

        paper = PaperGenerator.get_paper(w, h, paper_style)
        ink_color = (242, 245, 240) if is_dark_board else (34, 38, 46)

        # Chuẩn bị bản Live Sketch chất lượng cao để dùng cho chế độ Phác Thảo Thực Tế
        inv = 255 - gray
        blur_inv = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch_small = cv2.divide(gray, 255 - blur_inv, scale=256)
        sketch_full = cv2.cvtColor(
            cv2.resize(sketch_small, (w, h), interpolation=cv2.INTER_CUBIC),
            cv2.COLOR_GRAY2BGR
        )

        return {
            "orig": frame_bgr.copy(),
            "orig_f": frame_bgr.astype(np.float32),
            "paper": paper,
            "paper_f": paper.astype(np.float32),
            "paper_style": paper_style,
            "is_dark_board": is_dark_board,
            "ink_color": ink_color,
            "line_pts": line_pts,
            "line_draw": line_draw,
            "line_w": line_w,
            "hatch_pts": hatch_pts,
            "hatch_draw": hatch_draw,
            "hatch_w": hatch_w,
            "color_pts": c_pts,
            "color_draw": c_draw,
            "color_w": c_w,
            "brush_w": brush_w,
            "sketch_full": sketch_full,
        }

    @staticmethod
    def _draw_step_range(
        canvas: np.ndarray,
        pts: np.ndarray,
        draw_flags: np.ndarray,
        widths: np.ndarray,
        start_step: int,
        end_step: int,
        color: Tuple[int, int, int],
        override_thickness: Optional[int] = None
    ):
        """Vẽ nhanh chỉ đoạn [start_step .. end_step] để hỗ trợ render tăng dần O(delta)."""
        n = len(pts)
        s0 = max(1, start_step)
        s1 = min(n - 1, end_step)
        if s1 < s0:
            return
        pts_i = np.rint(pts).astype(np.int32)
        for i in range(s0, s1 + 1):
            if draw_flags[i]:
                thick = override_thickness if override_thickness is not None else max(1, int(round(widths[i])))
                cv2.line(
                    canvas,
                    (int(pts_i[i - 1, 0]), int(pts_i[i - 1, 1])),
                    (int(pts_i[i, 0]), int(pts_i[i, 1])),
                    color,
                    thick,
                    cv2.LINE_AA
                )

    @classmethod
    def render_frame(
        cls,
        plan: dict,
        progress: float,
        hand_bgra: Optional[np.ndarray],
        kf: Keyframe,
        session_state: Optional[dict] = None
    ) -> np.ndarray:
        """
        Render mượt mà một frame ở tiến trình `progress` [0.0 .. 1.0].
        Hỗ trợ `session_state` để vẽ tăng dần siêu nhanh khi phát tuần tự hoặc xuất video.
        """
        orig = plan["orig"]
        h, w = orig.shape[:2]
        mode = kf.mode

        if mode in ("Phác Thảo Thực Tế", "Live Sketch"):
            return cls.render_live_sketch_plan(plan, progress, hand_bgra, kf)
        if mode == "Cọ Tô Nhanh":
            return cls.render_brush_reveal(plan, progress, hand_bgra, kf)

        with_color = (mode == "Vẽ Nét + Tô Màu")
        use_hatch = (mode == "Chỉ Vẽ Nét")

        # Tiến trình chuẩn hóa với giai đoạn vào tay (0..0.04) và rút tay + nghỉ ngắm (0.90..1.0)
        enter_end = 0.04
        exit_start = 0.90
        exit_end = 0.955

        # Áp dụng easing cho tiến trình vẽ chính
        raw_draw_t = np.clip((progress - enter_end) / max(1e-5, exit_start - enter_end), 0.0, 1.0)
        if kf.easing == "Tự nhiên (Ease)":
            draw_t = _ease_in_out_sine(float(raw_draw_t))
        else:
            draw_t = float(raw_draw_t)

        line_pts = plan["line_pts"]
        line_draw = plan["line_draw"]
        line_w = plan["line_w"]
        n_line = len(line_pts)

        if with_color:
            line_phase_end = 0.54
            line_t = min(1.0, draw_t / line_phase_end)
            color_t = max(0.0, (draw_t - line_phase_end) / (1.0 - line_phase_end))
            hatch_t = 0.0
        else:
            # Chỉ Vẽ Nét: 76% vẽ nét chính, 24% đánh bóng chéo vùng tối
            has_hatch = use_hatch and len(plan["hatch_pts"]) > 4
            line_phase_end = 0.76 if has_hatch else 1.0
            line_t = min(1.0, draw_t / line_phase_end)
            hatch_t = max(0.0, (draw_t - line_phase_end) / max(1e-5, 1.0 - line_phase_end)) if has_hatch else 0.0
            color_t = 0.0

        target_line = int(round((n_line - 1) * line_t))
        target_hatch = int(round((len(plan["hatch_pts"]) - 1) * hatch_t)) if use_hatch else 0
        target_color = int(round((len(plan["color_pts"]) - 1) * color_t)) if with_color else 0

        # Quản lý state tăng dần (nếu tua lùi thì khởi tạo lại, nếu tiến tới thì chỉ vẽ thêm phần mới)
        if session_state is None:
            session_state = {}

        if (
            session_state.get("plan_id") != id(plan)
            or session_state.get("mode") != mode
            or target_line < session_state.get("cur_line", 0)
            or target_hatch < session_state.get("cur_hatch", 0)
            or target_color < session_state.get("cur_color", 0)
        ):
            session_state["plan_id"] = id(plan)
            session_state["mode"] = mode
            session_state["line_canvas"] = plan["paper"].copy()
            session_state["color_mask"] = np.zeros((h, w), dtype=np.uint8)
            session_state["cur_line"] = 0
            session_state["cur_hatch"] = 0
            session_state["cur_color"] = 0

        line_canvas = session_state["line_canvas"]
        color_mask = session_state["color_mask"]

        if target_line > session_state["cur_line"]:
            cls._draw_step_range(
                line_canvas, line_pts, line_draw, line_w,
                session_state["cur_line"] + 1, target_line,
                plan["ink_color"]
            )
            session_state["cur_line"] = target_line

        if use_hatch and target_hatch > session_state["cur_hatch"]:
            cls._draw_step_range(
                line_canvas, plan["hatch_pts"], plan["hatch_draw"], plan["hatch_w"],
                session_state["cur_hatch"] + 1, target_hatch,
                plan["ink_color"], override_thickness=1
            )
            session_state["cur_hatch"] = target_hatch

        if with_color and target_color > session_state["cur_color"]:
            cls._draw_step_range(
                color_mask, plan["color_pts"], plan["color_draw"], plan["color_w"],
                session_state["cur_color"] + 1, target_color,
                255, override_thickness=plan["brush_w"]
            )
            session_state["cur_color"] = target_color

        # Tổng hợp canvas kết quả bằng cv2.blendLinear SIMD (tốc độ >50 FPS)
        if with_color and color_t > 0.0:
            sw, sh = max(32, w // 4), max(32, h // 4)
            mask_small = cv2.resize(color_mask, (sw, sh), interpolation=cv2.INTER_AREA)
            mask_small = cv2.GaussianBlur(mask_small, (5, 5), 0)
            soft_u8 = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_LINEAR)
            w_orig = np.ascontiguousarray(soft_u8.astype(np.float32) * (1.0 / 255.0))
            w_line = np.ascontiguousarray(1.0 - w_orig)

            canvas = cv2.blendLinear(orig, line_canvas, w_orig, w_line)

            if not plan["is_dark_board"]:
                preserve_ratio = max(0.0, 1.0 - max(0.0, (draw_t - 0.85) / 0.15))
                if preserve_ratio > 0.02:
                    ink_preserve = cv2.min(canvas, line_canvas)
                    canvas = cv2.addWeighted(ink_preserve, preserve_ratio, canvas, 1.0 - preserve_ratio, 0)

            if progress >= 0.86:
                fin_a = _ease_in_out_sine(min(1.0, (progress - 0.86) / (exit_end - 0.86)))
                canvas = cv2.addWeighted(canvas, 1.0 - fin_a, orig, fin_a, 0)
        else:
            canvas = line_canvas.copy()

        # Tính toán vị trí đầu bút, độ nghiêng và trạng thái nhấc bút
        if hand_bgra is None or progress >= exit_end:
            return canvas

        entry_origin = np.array([w * 0.88, h * 1.12], dtype=np.float32)
        exit_dest = np.array([w * 0.92, h * 1.14], dtype=np.float32)

        # Xác định điểm đầu bút mục tiêu hiện tại
        if with_color and draw_t > 0.54 and len(plan["color_pts"]) > 1:
            active_pts = plan["color_pts"]
            active_draw = plan["color_draw"]
            idx_cur = min(len(active_pts) - 1, target_color)
        elif use_hatch and hatch_t > 0.0 and len(plan["hatch_pts"]) > 1:
            active_pts = plan["hatch_pts"]
            active_draw = plan["hatch_draw"]
            idx_cur = min(len(active_pts) - 1, target_hatch)
        else:
            active_pts = line_pts
            active_draw = line_draw
            idx_cur = min(len(active_pts) - 1, target_line)

        tip_xy = active_pts[idx_cur].copy()
        is_lifting = not bool(active_draw[idx_cur])

        # Tính vận tốc ngang để làm bàn tay nghiêng nhẹ tự nhiên
        prev_idx = max(0, idx_cur - 3)
        vx = float(active_pts[idx_cur, 0] - active_pts[prev_idx, 0])
        tilt_deg = float(np.clip(-vx * 0.16, -4.5, 4.5)) if kf.enable_tilt else 0.0
        lift_amt = 0.65 if is_lifting else 0.0

        # Chuyển động tay vào khung ở đầu (0 .. enter_end)
        if progress < enter_end:
            t_in = _ease_out_cubic(progress / enter_end)
            first_pt = line_pts[0]
            tip_xy = entry_origin * (1.0 - t_in) + first_pt * t_in
            lift_amt = 1.0 - t_in
            tilt_deg *= t_in
        # Chuyển động tay rút khỏi khung ở cuối (exit_start .. exit_end)
        elif progress > exit_start:
            t_out = _ease_in_cubic((progress - exit_start) / max(1e-5, exit_end - exit_start))
            tip_xy = tip_xy * (1.0 - t_out) + exit_dest * t_out
            lift_amt = t_out
            tilt_deg *= (1.0 - t_out)

        return HandCompositor.composite(
            canvas, hand_bgra,
            float(tip_xy[0]), float(tip_xy[1]),
            kf.tip_x, kf.tip_y, kf.hand_scale,
            tilt_deg=tilt_deg, lift_amount=lift_amt, enable_shadow=kf.enable_shadow
        )

    @classmethod
    def render_live_sketch_plan(
        cls,
        plan: dict,
        progress: float,
        hand_bgra: Optional[np.ndarray],
        kf: Keyframe
    ) -> np.ndarray:
        """
        Chế độ Phác Thảo Thực Tế: Bàn tay lướt nét cọ zích-zắc mở bản phác thảo chì rồi chuyển sang ảnh màu gốc.
        """
        orig = plan["orig"]
        sketch = plan["sketch_full"]
        paper = plan["paper"]
        h, w = orig.shape[:2]

        # Nét cọ quét zích-zắc qua trang giấy
        n_strokes = 9
        pts = []
        for i in range(n_strokes + 1):
            y = h * (0.07 + 0.86 * (i / n_strokes))
            x = w * (0.06 if i % 2 == 0 else 0.94)
            pts.append([x, y])
        sweep_pts = _resample_arc_length(_chaikin_smooth(np.asarray(pts, dtype=np.float32), 2), step_px=10.0)

        # Giai đoạn 1 (0..0.48): quét nét cọ hiện bản phác thảo chì; Giai đoạn 2 (0.48..0.92): chuyển sang ảnh gốc
        if progress <= 0.48:
            t = _ease_in_out_sine(progress / 0.48)
            idx = int(round((len(sweep_pts) - 1) * t))
            mask = np.zeros((h, w), dtype=np.uint8)
            bw = max(56, int(min(w, h) * 0.23))
            if idx >= 1:
                cv2.polylines(mask, [np.rint(sweep_pts[:idx + 1]).astype(np.int32)], False, 255, bw, cv2.LINE_AA)
            sw, sh = max(32, w // 4), max(32, h // 4)
            soft = cv2.resize(cv2.GaussianBlur(cv2.resize(mask, (sw, sh)), (7, 7), 0), (w, h))
            w1 = np.ascontiguousarray(soft.astype(np.float32) * (1.0 / 255.0))
            out = cv2.blendLinear(sketch, paper, w1, np.ascontiguousarray(1.0 - w1))
        else:
            t = _ease_in_out_sine(min(1.0, (progress - 0.48) / 0.42))
            idx = int(round((len(sweep_pts) - 1) * t))
            mask = np.zeros((h, w), dtype=np.uint8)
            bw = max(62, int(min(w, h) * 0.25))
            if idx >= 1:
                cv2.polylines(mask, [np.rint(sweep_pts[:idx + 1]).astype(np.int32)], False, 255, bw, cv2.LINE_AA)
            sw, sh = max(32, w // 4), max(32, h // 4)
            soft = cv2.resize(cv2.GaussianBlur(cv2.resize(mask, (sw, sh)), (7, 7), 0), (w, h))
            w1 = np.ascontiguousarray(soft.astype(np.float32) * (1.0 / 255.0))
            out = cv2.blendLinear(orig, sketch, w1, np.ascontiguousarray(1.0 - w1))
            if progress >= 0.86:
                fin_a = _ease_in_out_sine(min(1.0, (progress - 0.86) / 0.08))
                out = cv2.addWeighted(out, 1.0 - fin_a, orig, fin_a, 0)

        if hand_bgra is None or progress >= 0.95:
            return out

        tip = sweep_pts[idx].copy()
        prev = sweep_pts[max(0, idx - 2)]
        tilt = float(np.clip(-(tip[0] - prev[0]) * 0.14, -4.5, 4.5)) if kf.enable_tilt else 0.0
        lift = 0.0
        if progress > 0.89:
            t_out = _ease_in_cubic((progress - 0.89) / 0.06)
            tip[0] = tip[0] * (1.0 - t_out) + (w * 0.92) * t_out
            tip[1] = tip[1] * (1.0 - t_out) + (h * 1.12) * t_out
            lift = t_out

        return HandCompositor.composite(
            out, hand_bgra, float(tip[0]), float(tip[1]), kf.tip_x, kf.tip_y,
            kf.hand_scale, tilt_deg=tilt, lift_amount=lift, enable_shadow=kf.enable_shadow
        )

    @classmethod
    def render_brush_reveal(
        cls,
        plan: dict,
        progress: float,
        hand_bgra: Optional[np.ndarray],
        kf: Keyframe
    ) -> np.ndarray:
        """
        Chế độ Cọ Tô Nhanh: Nét cọ lớn quét zích-zắc mượt mà trực tiếp trên nền giấy trắng để mở ra khung hình.
        """
        orig = plan["orig"]
        paper = plan["paper"]
        h, w = orig.shape[:2]

        # Tạo quỹ đạo zích-zắc phủ toàn khung hình
        n_strokes = 9
        pts = []
        for i in range(n_strokes + 1):
            y = h * (0.07 + 0.86 * (i / n_strokes))
            x = w * (0.06 if i % 2 == 0 else 0.94)
            pts.append([x, y])
        sweep_pts = _resample_arc_length(_chaikin_smooth(np.asarray(pts, dtype=np.float32), 2), step_px=10.0)

        t = _ease_in_out_sine(min(1.0, progress / 0.90))
        idx = int(round((len(sweep_pts) - 1) * t))

        mask = np.zeros((h, w), dtype=np.uint8)
        bw = max(60, int(min(w, h) * 0.24))
        if idx >= 1:
            cv2.polylines(mask, [np.rint(sweep_pts[:idx + 1]).astype(np.int32)], False, 255, bw, cv2.LINE_AA)
        sw, sh = max(32, w // 4), max(32, h // 4)
        soft = cv2.resize(cv2.GaussianBlur(cv2.resize(mask, (sw, sh)), (9, 9), 0), (w, h))
        w1 = np.ascontiguousarray(soft.astype(np.float32) * (1.0 / 255.0))
        out = cv2.blendLinear(orig, paper, w1, np.ascontiguousarray(1.0 - w1))

        if progress >= 0.86:
            fin_a = _ease_in_out_sine(min(1.0, (progress - 0.86) / 0.08))
            out = cv2.addWeighted(out, 1.0 - fin_a, orig, fin_a, 0)

        if hand_bgra is None or progress >= 0.95:
            return out

        tip = sweep_pts[idx].copy()
        prev = sweep_pts[max(0, idx - 2)]
        tilt = float(np.clip(-(tip[0] - prev[0]) * 0.14, -4.5, 4.5)) if kf.enable_tilt else 0.0
        lift = 0.0
        if progress > 0.89:
            t_out = _ease_in_cubic((progress - 0.89) / 0.06)
            tip[0] = tip[0] * (1.0 - t_out) + (w * 0.92) * t_out
            tip[1] = tip[1] * (1.0 - t_out) + (h * 1.12) * t_out
            lift = t_out

        return HandCompositor.composite(
            out, hand_bgra, float(tip[0]), float(tip[1]), kf.tip_x, kf.tip_y,
            kf.hand_scale, tilt_deg=tilt, lift_amount=lift, enable_shadow=kf.enable_shadow
        )
