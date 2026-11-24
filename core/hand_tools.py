import json
import os
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class HandPreset:
    id: str
    name: str
    category: str
    filename: str
    tip_x: int
    tip_y: int
    default_scale: float = 0.65
    description: str = ""


def imread_unicode(path: str, flags: int = cv2.IMREAD_UNCHANGED) -> Optional[np.ndarray]:
    """Đọc ảnh hỗ trợ đường dẫn Unicode/tiếng Việt trên Windows."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def imwrite_unicode(path: str, img: np.ndarray) -> bool:
    """Lưu ảnh hỗ trợ đường dẫn Unicode/tiếng Việt trên Windows."""
    try:
        ext = os.path.splitext(path)[1] or ".png"
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            return False
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        buf.tofile(path)
        return True
    except Exception:
        return False


def _measure_border_key(bgr: np.ndarray) -> np.ndarray:
    """Đo màu nền chủ đạo từ viền ngoài ảnh."""
    h, w = bgr.shape[:2]
    b = max(4, min(16, min(h, w) // 25))
    samples = np.concatenate([
        bgr[:b, :, :].reshape(-1, 3),
        bgr[-b:, :, :].reshape(-1, 3),
        bgr[:, :b, :].reshape(-1, 3),
        bgr[:, -b:, :].reshape(-1, 3),
    ], axis=0)
    return np.median(samples, axis=0).astype(np.float32)


def remove_solid_background(
    img: np.ndarray,
    tolerance: float = 34.0,
    feather: float = 18.0,
    erode_px: int = 0
) -> np.ndarray:
    """
    Tách nền đồng nhất (Chroma Green / Magenta / Trắng / Xám) bằng đo màu viền,
    flood-fill ngoài biên để bảo toàn vùng trắng bên trong chủ thể, và khử viền màu (edge unmixing).
    """
    if img is None:
        raise ValueError("Input image is None")

    if len(img.shape) == 3 and img.shape[2] == 4:
        # Nếu ảnh đã có alpha rõ ràng ở viền, giữ nguyên alpha nếu đã tách nền sẵn
        alpha_border = np.concatenate([
            img[0, :, 3], img[-1, :, 3], img[:, 0, 3], img[:, -1, 3]
        ])
        if np.mean(alpha_border < 20) > 0.7:
            return img.copy()
        bgr = img[:, :, :3].copy()
    elif len(img.shape) == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        bgr = img[:, :, :3].copy()

    h, w = bgr.shape[:2]
    key = _measure_border_key(bgr)

    # Tính khoảng cách màu theo không gian Lab + BGR để ổn định với sáng/tối
    bgr_f = bgr.astype(np.float32)
    diff_bgr = np.linalg.norm(bgr_f - key[None, None, :], axis=2)

    # Nếu nền là chroma green mạnh (kênh G vượt trội R & B)
    is_green_chroma = (key[1] > key[2] + 60) and (key[1] > key[0] + 60)
    is_magenta_chroma = (key[2] > key[1] + 60) and (key[0] > key[1] + 60)

    if is_green_chroma:
        # Green excess metric giúp giữ màu da/gỗ/đen cực sạch
        green_excess = bgr_f[:, :, 1] - np.maximum(bgr_f[:, :, 0], bgr_f[:, :, 2])
        key_excess = float(key[1] - max(key[0], key[2]))
        dist = np.maximum(0.0, key_excess - green_excess)
        # Kết hợp khoảng cách BGR ở vùng tối/nhạt
        bg_candidate = (green_excess > key_excess * 0.48) | (diff_bgr < tolerance)
    elif is_magenta_chroma:
        mag_excess = np.minimum(bgr_f[:, :, 0], bgr_f[:, :, 2]) - bgr_f[:, :, 1]
        key_excess = float(min(key[0], key[2]) - key[1])
        dist = np.maximum(0.0, key_excess - mag_excess)
        bg_candidate = (mag_excess > key_excess * 0.48) | (diff_bgr < tolerance)
    else:
        dist = diff_bgr
        bg_candidate = diff_bgr < tolerance

    # Với nền chroma thuần (green/magenta), khử cả các khe nền khép kín giữa ngón tay và thân bút;
    # với nền trắng/xám thông thường, chỉ flood-fill từ 4 cạnh để không làm thủng vùng trắng của bút/tay áo.
    if is_green_chroma or is_magenta_chroma:
        ext_bg_mask = bg_candidate
    else:
        bg_u8 = (bg_candidate.astype(np.uint8) * 255)
        _, labels = cv2.connectedComponents(bg_u8, connectivity=8)
        border_labels = set()
        border_labels.update(np.unique(labels[0, :]).tolist())
        border_labels.update(np.unique(labels[-1, :]).tolist())
        border_labels.update(np.unique(labels[:, 0]).tolist())
        border_labels.update(np.unique(labels[:, -1]).tolist())
        border_labels.discard(0)
        ext_bg_mask = np.isin(labels, list(border_labels))

    # Tạo alpha mượt tại vùng chuyển tiếp cạnh
    alpha = np.ones((h, w), dtype=np.float32)
    alpha[ext_bg_mask] = 0.0

    # Vùng biên chuyển tiếp (dilated exterior background giao với foreground)
    ext_u8 = ext_bg_mask.astype(np.uint8) * 255
    if erode_px > 0:
        k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1))
        fg_u8 = cv2.erode(255 - ext_u8, k_erode)
        ext_u8 = 255 - fg_u8
        alpha[ext_u8 > 0] = 0.0

    dil_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edge_band = (cv2.dilate(ext_u8, dil_k) > 0) & (~ext_bg_mask)

    if is_green_chroma:
        # Tính alpha liên tục trên dải biên bằng độ dư green
        g_ex = bgr_f[:, :, 1] - np.maximum(bgr_f[:, :, 0], bgr_f[:, :, 2])
        key_ex = max(1.0, float(key[1] - max(key[0], key[2])))
        edge_a = np.clip(1.0 - (g_ex / (key_ex * 0.72)), 0.0, 1.0)
        alpha[edge_band] = edge_a[edge_band]
    elif is_magenta_chroma:
        m_ex = np.minimum(bgr_f[:, :, 0], bgr_f[:, :, 2]) - bgr_f[:, :, 1]
        key_ex = max(1.0, float(min(key[0], key[2]) - key[1]))
        edge_a = np.clip(1.0 - (m_ex / (key_ex * 0.72)), 0.0, 1.0)
        alpha[edge_band] = edge_a[edge_band]
    else:
        edge_a = np.clip((dist - tolerance * 0.8) / max(1.0, feather), 0.0, 1.0)
        alpha[edge_band] = edge_a[edge_band]

    # Làm mịn nhẹ alpha ở biên
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0.6)
    alpha[ext_bg_mask] = 0.0

    # Edge color unmixing: F = (C - (1 - a) * K) / a
    unmixed = bgr_f.copy()
    semi = (alpha > 0.04) & (alpha < 0.98)
    if np.any(semi):
        a_semi = alpha[semi][:, None]
        fg_est = (bgr_f[semi] - (1.0 - a_semi) * key[None, :]) / np.maximum(a_semi, 0.08)
        unmixed[semi] = np.clip(fg_est, 0, 255)

    # Khử ám xanh/tím còn sót trên viền bán trong suốt
    if is_green_chroma:
        spill_mask = (alpha > 0.02) & (alpha < 0.99)
        max_rb = np.maximum(unmixed[:, :, 0], unmixed[:, :, 2])
        excess = np.maximum(0.0, unmixed[:, :, 1] - max_rb)
        unmixed[:, :, 1] = np.where(spill_mask, unmixed[:, :, 1] - excess * 0.85, unmixed[:, :, 1])
    elif is_magenta_chroma:
        spill_mask = (alpha > 0.02) & (alpha < 0.99)
        g = unmixed[:, :, 1]
        unmixed[:, :, 0] = np.where(spill_mask & (unmixed[:, :, 0] > g), g + (unmixed[:, :, 0] - g) * 0.25, unmixed[:, :, 0])
        unmixed[:, :, 2] = np.where(spill_mask & (unmixed[:, :, 2] > g), g + (unmixed[:, :, 2] - g) * 0.25, unmixed[:, :, 2])

    # Xóa màu dưới pixel trong suốt hoàn toàn để resize không bị viền bẩn
    unmixed[alpha <= 0.02] = 0.0

    bgra = np.zeros((h, w, 4), dtype=np.uint8)
    bgra[:, :, :3] = np.clip(unmixed, 0, 255).astype(np.uint8)
    bgra[:, :, 3] = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    return bgra


def trim_and_pad_hand(bgra: np.ndarray, target_max_dim: int = 560, pad: int = 18) -> np.ndarray:
    """Cắt sát vùng có pixel hiển thị và chuẩn hóa kích thước với lề an toàn."""
    alpha = bgra[:, :, 3]
    ys, xs = np.where(alpha > 16)
    if len(xs) == 0:
        return bgra
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    cropped = bgra[y1:y2, x1:x2]

    ch, cw = cropped.shape[:2]
    scale = min(1.0, (target_max_dim - 2 * pad) / max(cw, ch))
    if scale < 1.0:
        nw, nh = max(16, int(cw * scale)), max(16, int(ch * scale))
        cropped = cv2.resize(cropped, (nw, nh), interpolation=cv2.INTER_AREA)
    ch, cw = cropped.shape[:2]
    out = np.zeros((ch + 2 * pad, cw + 2 * pad, 4), dtype=np.uint8)
    out[pad:pad + ch, pad:pad + cw] = cropped

    # Làm mềm mượt mép cắt của cổ tay áo ở góc dưới/phải (tránh vệt cắt ngang sắc cạnh khi tay bay cao)
    oh, ow = out.shape[:2]
    a = out[:, :, 3].astype(np.float32)
    fade_len = 28.0
    yy, xx = np.mgrid[0:oh, 0:ow].astype(np.float32)
    # Chỉ làm mờ dần khi chạm sát mép cắt dưới hoặc mép cắt phải của tay áo (nửa dưới-phải)
    bottom_edge_touch = np.any(a[oh - pad - 3:oh - pad + 1, int(ow * 0.35):] > 80)
    right_edge_touch = np.any(a[int(oh * 0.35):, ow - pad - 3:ow - pad + 1] > 80)
    fade_mask = np.ones((oh, ow), dtype=np.float32)
    if bottom_edge_touch:
        dist_b = (oh - pad - 1) - yy
        in_zone = (xx > ow * 0.30) & (dist_b < fade_len)
        fade_b = np.clip(dist_b / fade_len, 0.0, 1.0)
        fade_mask = np.where(in_zone, np.minimum(fade_mask, fade_b), fade_mask)
    if right_edge_touch:
        dist_r = (ow - pad - 1) - xx
        in_zone = (yy > oh * 0.30) & (dist_r < fade_len)
        fade_r = np.clip(dist_r / fade_len, 0.0, 1.0)
        fade_mask = np.where(in_zone, np.minimum(fade_mask, fade_r), fade_mask)

    out[:, :, 3] = np.clip(a * fade_mask, 0, 255).astype(np.uint8)
    return out


def detect_pen_tip(bgra: np.ndarray) -> Tuple[int, int]:
    """
    Tự động ước lượng chính xác tọa độ đầu bút từ ảnh bàn tay BGRA:
    kiểm tra phía mảnh hơn (đầu bút vs cổ tay) rồi lấy điểm cực trị sát đầu bút nhất.
    """
    alpha = bgra[:, :, 3]
    ys, xs = np.where(alpha > 110)
    if len(xs) == 0:
        return (20, 20)

    min_x, max_x = int(xs.min()), int(xs.max())
    # Xác định đầu bút nằm bên trái hay bên phải dựa trên độ dày vùng gần biên trái vs phải
    span = max(20, max_x - min_x)
    left_band = ys[xs <= min_x + max(12, span // 8)]
    right_band = ys[xs >= max_x - max(12, span // 8)]
    left_spread = int(left_band.max() - left_band.min()) if len(left_band) else 9999
    right_spread = int(right_band.max() - right_band.min()) if len(right_band) else 9999

    if left_spread <= right_spread:
        tip_band_y = ys[xs <= min_x + 7]
        tx = min_x + 2
        ty = int(np.median(tip_band_y)) if len(tip_band_y) else int(ys[np.argmin(xs)])
    else:
        tip_band_y = ys[xs >= max_x - 7]
        tx = max_x - 2
        ty = int(np.median(tip_band_y)) if len(tip_band_y) else int(ys[np.argmax(xs)])
    return (tx, ty)


def build_checkerboard(w: int, h: int, cell: int = 10) -> np.ndarray:
    """Tạo nền ca-rô biểu thị vùng trong suốt."""
    yy, xx = np.indices((h, w))
    grid = ((xx // cell) + (yy // cell)) % 2
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[grid == 0] = (46, 51, 64)
    img[grid == 1] = (34, 38, 48)
    return img


def load_hand_presets(hands_dir: str) -> List[HandPreset]:
    meta_path = os.path.join(hands_dir, "presets.json")
    if not os.path.exists(meta_path):
        return []
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return [HandPreset(**item) for item in raw]
    except Exception:
        return []


def save_hand_presets(hands_dir: str, presets: List[HandPreset]) -> None:
    os.makedirs(hands_dir, exist_ok=True)
    meta_path = os.path.join(hands_dir, "presets.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump([asdict(p) for p in presets], f, ensure_ascii=False, indent=2)
