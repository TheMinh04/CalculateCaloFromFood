# Hướng dẫn train CalcuCalo trên Kaggle từ đầu đến cuối

Tài liệu này mô tả đúng thứ tự các cell cần chạy để train detector VietFood67, export ONNX và lưu kết quả an toàn.

## 1. Nguyên tắc quan trọng

- Các cell **có liên quan với nhau** và phải chạy từ trên xuống dưới.
- `/kaggle/input` chứa dataset đã gắn vào notebook và chỉ đọc.
- `/kaggle/working` chứa source code, checkpoint và kết quả được tạo trong lúc chạy.
- `Files only` là persistence theo kiểu best-effort, không phải bản backup được bảo đảm.
- Source code mất có thể clone lại từ GitHub; checkpoint mất thì phải train lại.
- Sau khi train, phải tải file ZIP về máy hoặc tạo một Saved Version có Output trước khi restart/tắt session.
- Không bấm Restart, Factory Reset, đổi Accelerator hoặc đổi Environment khi đang train hay trước khi sao lưu kết quả.

Quy trình khuyến nghị cho lần chạy chính thức là chuẩn bị notebook chạy được từ đầu đến cuối, sau đó chọn **Save Version → Save & Run All**. Kaggle sẽ chạy notebook trong một session sạch và lưu Output của version khi toàn bộ notebook hoàn tất.

## 2. Cấu hình notebook trước khi chạy

Trong bảng **Session options**:

1. Chọn `Accelerator: GPU T4 x2` nếu tài khoản đang được cấp hai GPU.
2. Bật `Internet` để clone GitHub và tải package/model pretrained.
3. Có thể chọn `Persistence: Files only` hoặc `Variables and Files`.
4. Gắn dataset `VietFood67` bằng **Add Input**.

Lưu ý: thay đổi Accelerator, Environment hoặc Persistence có thể làm Kaggle tạo session mới. Hãy cấu hình xong trước khi chạy các cell bên dưới.

## 3. Cell 1 — Kiểm tra GPU

```python
!nvidia-smi

import torch

print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("Số GPU:", torch.cuda.device_count())

assert torch.cuda.is_available(), "CUDA chưa hoạt động. Hãy bật GPU trong Session options."

DEVICE = "0,1" if torch.cuda.device_count() >= 2 else "0"
print("Thiết bị sẽ dùng:", DEVICE)
```

Kết quả mong đợi:

- `CUDA: True`.
- `Số GPU: 2` nếu dùng T4 x2.
- `DEVICE` là `0,1` với hai GPU hoặc `0` với một GPU.

## 4. Cell 2 — Clone hoặc cập nhật source code

Cell này có thể chạy lại. Nếu project chưa tồn tại thì clone; nếu đã tồn tại thì cập nhật bằng `git pull`.

```python
!if [ -d /kaggle/working/CalcuCalo/.git ]; then \
    git -C /kaggle/working/CalcuCalo pull --ff-only; \
  else \
    git clone https://github.com/TheMinh04/CalculateCaloFromFood.git /kaggle/working/CalcuCalo; \
  fi

%cd /kaggle/working/CalcuCalo
```

Kiểm tra:

```python
!git status --short
!git log -1 --oneline
```

## 5. Cell 3 — Cài thư viện bằng đúng Python của kernel

```python
import sys

!{sys.executable} -m pip install -q \
  -e "/kaggle/working/CalcuCalo[inference]" \
  onnx \
  onnxslim
```

Không cần restart session sau cell này.

## 6. Cell 4 — Đưa source vào Python path và kiểm tra import

Editable install đôi khi chưa được kernel hiện tại nhận ra ngay. Cell này thêm trực tiếp thư mục `src` mà không cần restart.

```python
import sys

SRC_PATH = "/kaggle/working/CalcuCalo/src"
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import calcucalo
import ultralytics

print("CalcuCalo:", calcucalo.__version__)
print("Ultralytics:", ultralytics.__version__)
```

Kết quả mong đợi có dạng:

```text
CalcuCalo: 0.3.0
Ultralytics: 8.x.x
```

## 7. Cell 5 — Tự tìm đường dẫn dataset

Không viết cứng `/kaggle/input/vietfood68/...` vì đường dẫn mount có thể thay đổi giữa giao diện hoặc phiên bản Kaggle.

```python
from pathlib import Path

input_root = Path("/kaggle/input")
candidate_roots = set()

for images_train in input_root.rglob("images/train"):
    root = images_train.parent.parent
    if (root / "labels" / "train").is_dir():
        candidate_roots.add(root)

for train_images in input_root.rglob("train/images"):
    root = train_images.parent.parent
    if (root / "train" / "labels").is_dir():
        candidate_roots.add(root)

candidates = sorted(candidate_roots)

print("Các dataset YOLO tìm thấy:")
for candidate in candidates:
    print("-", candidate)

assert candidates, "Không tìm thấy cấu trúc images/train + labels/train trong /kaggle/input."

DATASET_ROOT = candidates[0]
print("Dataset sẽ sử dụng:", DATASET_ROOT)
```

Với dataset hiện tại, kết quả thường là:

```text
/kaggle/input/datasets/thomasnguyen6868/vietfood68/dataset
```

Nếu có nhiều hơn một candidate, hãy kiểm tra danh sách và gán `DATASET_ROOT` tới đúng thư mục VietFood67.

## 8. Cell 6 — Tạo và kiểm tra `vietfood67.yaml`

```python
import os
import subprocess
import sys
from pathlib import Path

DATA_YAML = Path("/kaggle/working/vietfood67.yaml")

env = os.environ.copy()
env["PYTHONPATH"] = "/kaggle/working/CalcuCalo/src"

subprocess.run(
    [
        sys.executable,
        "-m",
        "calcucalo.cli",
        "prepare-dataset",
        str(DATASET_ROOT),
        "--output",
        str(DATA_YAML),
    ],
    check=True,
    env=env,
)

assert DATA_YAML.is_file(), f"Không tạo được {DATA_YAML}"
print(DATA_YAML.read_text(encoding="utf-8"))
```

File hợp lệ phải có các dòng tương tự:

```yaml
path: /kaggle/input/datasets/thomasnguyen6868/vietfood68/dataset
train: images/train
val: images/valid
test: images/test
names:
  0: Banh canh
  # ... đủ 68 lớp
```

Nếu cell báo `FileNotFoundError`, hãy quay lại Cell 5 và kiểm tra `DATASET_ROOT`.

## 9. Cell 7 — Smoke test tùy chọn

Smoke test chỉ dùng để xác nhận pipeline chạy được. Không dùng checkpoint của bước này làm model chính.

```python
SMOKE_NAME = "smoke_test"

!yolo detect train \
  data=/kaggle/working/vietfood67.yaml \
  model=yolo11n.pt \
  epochs=1 \
  imgsz=320 \
  batch=32 \
  device={DEVICE} \
  workers=4 \
  fraction=0.01 \
  val=False \
  project=/kaggle/working/runs/detect \
  name={SMOKE_NAME} \
  exist_ok=True
```

Có thể bỏ qua cell này nếu pipeline đã được smoke test thành công trước đó.

## 10. Cell 8 — Train model chính

Hướng dẫn dùng Ultralytics CLI trực tiếp để tránh khác biệt kiểu dữ liệu trả về của Python API khi train multi-GPU.

```python
RUN_NAME = "vietfood67_yolo11n_v1"
```

```python
!yolo detect train \
  data=/kaggle/working/vietfood67.yaml \
  model=yolo11n.pt \
  epochs=10 \
  imgsz=640 \
  batch=32 \
  device={DEVICE} \
  workers=4 \
  patience=10 \
  seed=42 \
  deterministic=True \
  plots=True \
  save_period=1 \
  project=/kaggle/working/runs/detect \
  name={RUN_NAME} \
  exist_ok=True
```

Giải thích:

- `epochs=10`: baseline đầu tiên; có thể tăng lên 30–50 sau khi đánh giá.
- `imgsz=640`: giữ chi tiết tốt hơn 320 px.
- `device={DEVICE}`: tự dùng hai GPU nếu Cell 1 phát hiện hai GPU.
- `save_period=1`: lưu checkpoint theo từng epoch trong session hiện tại.
- `exist_ok=True`: giữ đúng tên thư mục; hãy đổi `RUN_NAME` khi muốn tạo experiment mới.

Không chỉnh sửa hoặc chạy lại cell train khi cell đang hoạt động.

## 11. Cell 9 — Kiểm tra checkpoint và metric

```python
from pathlib import Path
import pandas as pd

RUN_DIR = Path("/kaggle/working/runs/detect") / RUN_NAME
BEST_PT = RUN_DIR / "weights" / "best.pt"
LAST_PT = RUN_DIR / "weights" / "last.pt"
RESULTS_CSV = RUN_DIR / "results.csv"

assert BEST_PT.is_file(), f"Thiếu checkpoint: {BEST_PT}"
assert LAST_PT.is_file(), f"Thiếu checkpoint: {LAST_PT}"
assert RESULTS_CSV.is_file(), f"Thiếu metric: {RESULTS_CSV}"

print("Run directory:", RUN_DIR)
print("Best checkpoint:", BEST_PT, BEST_PT.stat().st_size, "bytes")
print("Last checkpoint:", LAST_PT, LAST_PT.stat().st_size, "bytes")

results = pd.read_csv(RESULTS_CSV)
display(results.tail(1).T)
```

Metric quan trọng:

- `metrics/precision(B)`
- `metrics/recall(B)`
- `metrics/mAP50(B)`
- `metrics/mAP50-95(B)`

## 12. Cell 10 — Export `best.pt` sang ONNX

```python
from ultralytics import YOLO

exported_onnx = YOLO(str(BEST_PT)).export(
    format="onnx",
    imgsz=640,
    dynamic=True,
    simplify=True,
    device=0,
)

BEST_ONNX = Path(exported_onnx)
assert BEST_ONNX.is_file(), f"Export ONNX thất bại: {BEST_ONNX}"

print("ONNX:", BEST_ONNX)
print("Kích thước:", BEST_ONNX.stat().st_size, "bytes")
```

ONNX thường được tạo tại:

```text
/kaggle/working/runs/detect/vietfood67_yolo11n_v1/weights/best.onnx
```

## 13. Cell 11 — Kiểm thử suy luận nhanh

```python
test_images = sorted((DATASET_ROOT / "images" / "test").glob("*"))
assert test_images, "Không tìm thấy ảnh test."

test_image = test_images[0]
print("Ảnh kiểm thử:", test_image)

test_model = YOLO(str(BEST_PT))
predictions = test_model.predict(
    source=str(test_image),
    imgsz=640,
    conf=0.25,
    device=0,
    save=True,
    project="/kaggle/working/predictions",
    name=RUN_NAME,
    exist_ok=True,
)

print("Số ảnh đã xử lý:", len(predictions))
print("Số detection:", len(predictions[0].boxes))
```

Nếu muốn xem ảnh kết quả:

```python
from IPython.display import Image, display

prediction_file = Path("/kaggle/working/predictions") / RUN_NAME / test_image.name
display(Image(filename=str(prediction_file)))
```

## 14. Cell 12 — Đóng gói toàn bộ artifact

```python
import shutil
from pathlib import Path

ARCHIVE_BASE = Path("/kaggle/working") / f"{RUN_NAME}_artifacts"
archive_path = shutil.make_archive(
    str(ARCHIVE_BASE),
    "zip",
    root_dir=str(RUN_DIR),
)

archive = Path(archive_path)
assert archive.is_file(), "Không tạo được file ZIP."

print("ZIP:", archive)
print("Kích thước:", archive.stat().st_size, "bytes")
```

File ZIP bao gồm checkpoint, ONNX, metric, biểu đồ và ảnh validation của run.

## 15. Cell 13 — Tải ZIP về máy

```python
from IPython.display import FileLink, display

display(FileLink(str(archive)))
```

Nhấn vào đường dẫn được hiển thị để tải ZIP về máy. Sau khi tải xong, kiểm tra file tồn tại trong thư mục Downloads trước khi tắt session.

## 16. Lưu Output lâu dài trên Kaggle

### Cách A — Khuyến nghị cho một lần chạy hoàn chỉnh

1. Bảo đảm notebook có đầy đủ Cell 1 đến Cell 12 theo đúng thứ tự.
2. Chọn **Save Version**.
3. Chọn **Save & Run All**.
4. Chờ toàn bộ notebook chạy hoàn tất.
5. Mở version vừa tạo và kiểm tra phần Output có file ZIP.

`Save & Run All` chạy notebook trong session sạch. Vì vậy notebook không được phụ thuộc vào file đã tạo thủ công từ một session trước.

### Cách B — Chạy tương tác

Nếu bấm Run từng cell, hãy tải ZIP về máy ngay sau Cell 13. Không chờ đến lúc session timeout và không coi `Files only` là bản backup duy nhất.

## 17. Tiếp tục một run bị gián đoạn

Chỉ có thể resume nếu `last.pt` vẫn còn và chứa trạng thái training phù hợp:

```python
LAST_PT = Path("/kaggle/working/runs/detect") / RUN_NAME / "weights" / "last.pt"
assert LAST_PT.is_file(), f"Không tìm thấy {LAST_PT}"
```

```python
!yolo detect train resume model={LAST_PT}
```

Nếu toàn bộ `/kaggle/working` đã mất và không có Saved Version, Kaggle Dataset hoặc file đã tải về thì không thể resume.

## 18. Xử lý lỗi thường gặp

### `ModuleNotFoundError: No module named 'calcucalo'`

Chạy lại Cell 4 để thêm `/kaggle/working/CalcuCalo/src` vào `sys.path`. Không cần Factory Reset.

### `Could not find YOLO train/val folders`

`DATASET_ROOT` đang sai. Chạy lại Cell 5 và dùng thư mục chứa trực tiếp `images/` cùng `labels/`.

### `AttributeError: 'dict' object has no attribute 'save_dir'`

Đây là lỗi tương thích của wrapper Python cũ sau khi train multi-GPU. Hướng dẫn này dùng `yolo detect train` trực tiếp nên không đi qua đoạn code đó.

### CUDA không hoạt động

Kiểm tra Accelerator. Nếu thay đổi Accelerator, Kaggle có thể tạo session mới; sau đó phải chạy lại từ Cell 1.

### Hết bộ nhớ GPU

Giảm batch:

```text
batch=16
```

Nếu vẫn thiếu bộ nhớ, thử `batch=8`.

### File trong `/kaggle/working` biến mất

Persistence có thể thất bại khi session crash hoặc bị thu hồi. Khôi phục từ một trong các nguồn sau:

1. File ZIP đã tải về máy.
2. Output của Saved Version.
3. Kaggle Dataset đã tạo từ artifact.

Nếu không có cả ba nguồn thì checkpoint không thể khôi phục và phải train lại.

## 19. Checklist hoàn thành

- [ ] CUDA hoạt động và số GPU đúng.
- [ ] Source code đã clone/cập nhật.
- [ ] `calcucalo` và `ultralytics` import thành công.
- [ ] `DATASET_ROOT` trỏ đúng thư mục chứa `images/` và `labels/`.
- [ ] `/kaggle/working/vietfood67.yaml` tồn tại và có đủ 68 lớp.
- [ ] Train kết thúc và có `weights/best.pt`.
- [ ] Có `weights/last.pt`.
- [ ] Có `results.csv` và các biểu đồ đánh giá.
- [ ] Export thành công `weights/best.onnx`.
- [ ] Kiểm thử suy luận ít nhất một ảnh thành công.
- [ ] Tạo file ZIP artifact.
- [ ] Đã tải ZIP về máy hoặc xác nhận Output của Saved Version.

## 20. Các file quan trọng cần giữ

Tối thiểu phải giữ:

```text
weights/best.pt
weights/last.pt
weights/best.onnx
results.csv
results.png
confusion_matrix.png
confusion_matrix_normalized.png
args.yaml
```

`best.pt` là checkpoint ưu tiên dùng cho suy luận. `last.pt` chủ yếu dùng để resume khi trạng thái optimizer còn được giữ. `best.onnx` dùng cho ONNX Runtime hoặc triển khai ngoài Ultralytics.

## Tham khảo Kaggle

- [Kaggle Notebooks documentation](https://www.kaggle.com/docs/notebooks)
- [Kaggle Staff: Session Persistence for Variables and Files](https://www.kaggle.com/discussions/product-feedback/355440)
