# CalcuCalo Vision

Baseline Python cho phần phân tích ảnh của ứng dụng dinh dưỡng món Việt:

1. YOLO nhận diện món ăn theo 68 lớp của VietFood67.
2. Mask có sẵn từ YOLO-seg được dùng trực tiếp; với model detection, bounding box được đưa vào SAM 2 hoặc GrabCut để bóc tách vùng món ăn.
3. Mask được đổi thành polygon để mobile app có thể vẽ và cho người dùng hiệu chỉnh.
4. Khối lượng được ước lượng bằng diện tích thật × độ dày × khối lượng riêng khi ảnh có tỷ lệ mét. Nếu ảnh không có vật chuẩn, hệ thống trả về serving prior cùng khoảng bất định rõ ràng.

Đây là baseline nghiên cứu, không phải thiết bị đo dinh dưỡng hay thiết bị y tế.

## Vì sao không train YOLO-seg trực tiếp?

VietFood67 có khoảng 33.003 ảnh, 67 lớp món ăn và một lớp người, nhưng nhãn công khai là **bounding box**, không phải polygon mask, depth hay khối lượng. Vì vậy:

- YOLO detect có thể train trực tiếp bằng dataset.
- Mask ở bản đầu được tạo bằng SAM 2 từ bounding box (chất lượng tốt hơn, cần tải thêm model) hoặc GrabCut (nhẹ, chạy CPU).
- Khối lượng từ một ảnh RGB không có vật chuẩn không thể là phép đo metric. Pipeline không biến relative monocular depth thành cm giả; nó yêu cầu `cm_per_pixel`, đường kính đĩa đã biết, hoặc trả serving prior để người dùng sửa.

Dataset nặng khoảng 26,4 GB và dùng giấy phép **CC BY-NC-SA 4.0**. Metadata của checkpoint ONNX tham khảo ghi giấy phép **AGPL-3.0**. Cần kiểm tra cả giấy phép dữ liệu lẫn model/framework trước khi làm sản phẩm thương mại.

## Cài đặt

Khuyến nghị Python 3.10–3.12 và môi trường ảo:

```powershell
cd D:\Model\CalcuCalo
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[all]"
```

Nếu chỉ chạy checkpoint ONNX trên CPU và không cần train/SAM, có thể cài bản nhẹ bằng `pip install -e ".[onnx]"`. CLI tự chọn ONNX Runtime cho file `.onnx` và Ultralytics cho file `.pt`.

## Chạy thử với trọng số tham khảo

Repo của tác giả có một checkpoint YOLOv10m ONNX để demo. Metadata bên trong checkpoint này cho thấy nó là bản cũ gồm **58 lớp (57 món + người)**, chưa phải đủ 68 lớp của VietFood67. Có thể dùng để kiểm tra pipeline ngay; để nhận diện đủ lớp vẫn cần train theo phần tiếp theo. Script tải model về local (file model bị `.gitignore`):

```powershell
python scripts/download_reference_weights.py
calcucalo analyze path\to\meal.jpg `
  --model models\vietfood57_yolov10m.onnx `
  --segmenter grabcut `
  --plate-diameter-cm 25 `
  --output-json outputs\result.json `
  --output-image outputs\overlay.jpg
```

Dùng SAM 2 để có mask tốt hơn (lần đầu Ultralytics sẽ tải trọng số):

```powershell
calcucalo analyze path\to\meal.jpg `
  --model models\vietfood57_yolov10m.onnx `
  --segmenter sam `
  --sam-model sam2.1_t.pt `
  --output-json outputs\result.json
```

Nếu app camera đã tự hiệu chuẩn, truyền `--cm-per-pixel 0.042` thay cho đường kính đĩa. Nếu không truyền tỷ lệ, `portion.method` sẽ là `single_image_serving_prior` hoặc `liquid_or_mixed_dish_prior` và confidence thấp hơn.

## Tải và chuẩn bị VietFood67

Cài Kaggle CLI, đăng nhập, sau đó tải dataset:

```powershell
pip install kaggle
kaggle datasets download -d thomasnguyen6868/vietfood68 --unzip -p data\raw\vietfood67
```

Kiểm tra cấu trúc, class ID, tọa độ nhãn và sinh file cấu hình YOLO:

```powershell
calcucalo prepare-dataset data\raw\vietfood67 --output configs\vietfood67.yaml
```

Tool nhận cả hai layout phổ biến:

```text
root/train/images + root/train/labels
root/images/train + root/labels/train
```

`valid` và `validation` được tự động ánh xạ thành `val`.

## Train detection model

Baseline cân bằng tốc độ/VRAM dùng YOLO11n; có thể đổi thành `yolo11s.pt` hoặc `yolo11m.pt` nếu GPU đủ mạnh:

```powershell
python scripts/train_detector.py `
  --data configs\vietfood67.yaml `
  --model yolo11n.pt `
  --epochs 100 `
  --batch -1 `
  --device 0 `
  --export-onnx
```

Checkpoint tốt nhất nằm tại `runs/detect/vietfood67_yolo11n/weights/best.pt`. Dataset gốc chỉ huấn luyện **detection**. Muốn có model segmentation thuần, cần tạo và kiểm tra polygon ground truth (có thể lấy SAM 2 làm pseudo-label rồi sửa bằng người) trước khi train `yolo11n-seg.pt`.

## API FastAPI

```powershell
$env:CALCUCALO_MODEL = "models\vietfood57_yolov10m.onnx"
$env:CALCUCALO_SEGMENTER = "grabcut" # hoặc sam
uvicorn calcucalo.api:app --host 0.0.0.0 --port 8000
```

Gọi API:

```powershell
curl.exe -X POST "http://localhost:8000/api/v1/food/analyze" `
  -F "image=@path\to\meal.jpg" `
  -F "plate_diameter_cm=25"
```

Response gồm `bbox_xyxy`, `mask_polygons`, `detection_confidence`, `weight_g`, khoảng ước lượng, phương pháp và các giả định. Endpoint health là `GET /health`.

## Hiệu chuẩn khối lượng cho dữ liệu thật

Các giá trị trong `configs/portion_priors.yaml` chỉ là điểm khởi đầu. Để model hữu dụng:

1. Chụp món từ góc gần vuông góc, luôn có đĩa/bát kích thước biết trước hoặc marker ArUco/card chuẩn.
2. Cân từng thành phần bằng cân bếp và lưu `true_weight_g` cùng ảnh/mask.
3. Fit lại `thickness_cm`, `density_g_cm3` hoặc một regressor theo class trên train set; đánh giá MAE/MAPE trên người dùng và quán ăn chưa xuất hiện trong train.
4. Với phở/bún/canh, thu thêm đường kính, chiều cao bát và mức nước hoặc dùng RGB-D/multi-view. Không suy thể tích nước từ diện tích mặt thoáng.
5. Luôn giữ bước hiệu chỉnh của người dùng; lưu correction để active learning.

## Kiểm thử

```powershell
pytest
```

Các test hiện kiểm tra công thức portion, fallback không hiệu chuẩn, lọc lớp người, serialization và validator của YOLO dataset.
