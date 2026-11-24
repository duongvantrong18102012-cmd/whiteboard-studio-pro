# Whiteboard Studio Pro

Công cụ tạo video vẽ tay 2D (Whiteboard Animation) viết bằng PyQt6 & OpenCV, hỗ trợ cả **Video đầu vào** lẫn **Chuỗi Ảnh ghép thành Video**.

## Cấu trúc dự án

- `main.py` — Điểm khởi chạy ứng dụng.
- `core/engine.py` — Engine phân tích nét khung xương Zhang-Suen, làm mượt Chaikin, lấy mẫu theo cung đều, nhấc bút có đường cong chuyển tiếp, tô màu theo trục chính (PCA) & ghép bàn tay 2D có bóng đổ mềm + nghiêng cổ tay (`cv2.blendLinear` SIMD > 60 FPS tại 720p).
- `core/hand_tools.py` — Tách nền đồng nhất (Chroma Green / Magenta / Trắng), khử viền màu, làm mềm mép cổ tay áo và tự động dò tọa độ đầu bút.
- `core/media.py` — Quản lý Video (đọc tuần tự + cache LRU) và Chuỗi Ảnh (đồng bộ thời lượng slide, chọn chất liệu giấy và khổ khung hình 16:9 / 9:16 / 1:1).
- `core/workers.py` — Phân tích cảnh và xuất video MP4 chuẩn H.264 (FFmpeg).
- `ui/` — Giao diện Studio gọn gàng với 3 tab (`Hiệu Ứng`, `Bàn Tay 2D`, `Dự Án & Xuất`), Timeline kéo thả Keyframe, và công cụ chấm tọa độ đầu bút trực quan.
- `assets/hands/` — 8 mẫu bàn tay 2D tách nền sẵn (`presets.json`).
- `tests/` — Bộ kiểm thử tự động cho engine, tách nền, xuất MP4 và giao diện.

## Chạy ứng dụng & Kiểm thử

```bash
pip install -r requirements.txt
python main.py
python -m unittest discover -s tests -v
```
