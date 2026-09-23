# Báo cáo trạng thái model CalcuCalo Vision

Ngày cập nhật: 21/09/2026  
Phiên bản mã nguồn: 0.3.0

## 1. Kết luận ngắn

Hệ thống hiện là một **baseline chạy được từ đầu đến cuối**, gồm nhận diện món, tạo mask, ước lượng khẩu phần, phân rã món phức hợp theo công thức và xuất JSON dinh dưỡng. Hệ thống phù hợp để làm MVP, thu thập dữ liệu và kiểm thử API.

Hệ thống **chưa phải model dinh dưỡng đã được kiểm định**. VietFood67 chỉ cung cấp bounding box, không cung cấp mask thành phần, độ sâu, khối lượng thật hay công thức của từng đĩa. Vì vậy không được hiểu các thành phần ẩn dưới nước dùng hoặc nước sốt là đã được model nhìn thấy. Code phân biệt rõ:

- `visual_match`: có detection thành phần nằm trong vùng món tổng.
- `visual_metric_estimate`: thành phần được phát hiện và ảnh có tỷ lệ mét để ước lượng khối lượng.
- `catalog_prior`: thành phần suy ra từ công thức chuẩn, chưa được nhìn thấy trực tiếp.

## 2. Những gì model đã có

| Khối | Trạng thái | Cách hoạt động |
|---|---|---|
| Tiền xử lý ảnh | Đã có | Đọc JPG/PNG/WebP, sửa EXIF orientation, chuẩn hóa RGB |
| Nhận diện món | Đã có | YOLO `.pt` qua Ultralytics hoặc `.onnx` qua ONNX Runtime |
| Model chạy thử | Đã có | Checkpoint YOLOv10m công khai, 58 lớp của bản dữ liệu cũ |
| Cấu hình VietFood67 | Đã có | Đủ 68 class ID, gồm 67 món và lớp người |
| Train model mới | Đã có code | Validator dataset, train YOLO11, resume, export ONNX |
| Bóc tách món | Đã có | Mask từ YOLO-seg; nếu là detection model thì dùng SAM 2 hoặc GrabCut |
| Polygon cho mobile | Đã có | Chuyển binary mask thành polygon JSON gọn |
| Ước lượng khối lượng | Đã có baseline | Diện tích mask × độ dày × khối lượng riêng khi có tỷ lệ mét |
| Ảnh không có vật chuẩn | Đã có fallback | Dùng khẩu phần cơ sở và trả khoảng bất định thấp |
| Món dạng nước | Đã xử lý an toàn | Không biến diện tích mặt nước thành thể tích giả; dùng prior |
| Phân rã món phức hợp | Đã có | Recipe catalog cho đủ 67 lớp món ăn |
| Component second pass | Đã có | Crop món phức hợp rồi detect lại; hỗ trợ checkpoint component riêng |
| Ghép detection con | Đã có | Detection `Com`, `Thit nuong`, `Cha/Trung` nằm trong `Com tam` được ghép thành component |
| Tính macro | Đã có | Tính calories, protein, fat, carb theo gram của từng component |
| JSON mobile | Đã có | Contract gọn `food_id/name/base_portion_g/components` |
| JSON nghiên cứu | Đã có | Thêm bbox, mask, confidence, estimated totals, basis và warnings |
| Calculation trace | Đã có | Trả công thức, đầu vào, giá trị trung gian, giới hạn, sai số và output cho calibration/portion/macro |
| User correction | Đã có | Override gram từng component và tính lại tổng macro |
| API | Đã có | FastAPI `POST /api/v1/food/analyze` |
| Contract validation | Đã có | JSON Schema cho nutrition response và evaluation manifest |
| Catalog QA | Đã có | Kiểm tra coverage, duplicate ID, calories/macro và tổng gram |
| End-to-end evaluation | Đã có | Precision/recall/F1 và MAE/MAPE cho weight/calories/macros |
| Test tự động | Đã có | Công thức portion, catalog, coverage 67 lớp, component grouping và schema |

## 3. Kiến trúc hiện tại

```text
Ảnh RGB
  └─ YOLO detect/segment
      ├─ nhãn món tổng: Com tam
      ├─ crop món tổng và chạy component pass ở độ phân giải cao hơn
      ├─ nhãn con nếu model thấy: Com, Thit nuong, Cha, Trung
      └─ bounding boxes / masks
          └─ SAM 2 hoặc GrabCut nếu chưa có mask
              └─ hiệu chuẩn cm/pixel bằng đĩa hoặc client camera
                  └─ ước lượng khối lượng + khoảng sai số
                      └─ recipe catalog
                          ├─ ghép component có bằng chứng thị giác
                          ├─ bổ sung component bị che bằng catalog prior
                          └─ tính calories và macro
                              └─ JSON full hoặc nutrition
```

Các file quan trọng:

- `src/calcucalo/detector.py`: YOLO ONNX/Ultralytics.
- `src/calcucalo/segmenter.py`: SAM 2, GrabCut và bbox fallback.
- `src/calcucalo/calibration.py`: tỷ lệ ảnh từ đường kính đĩa.
- `src/calcucalo/portion.py`: khối lượng và uncertainty.
- `src/calcucalo/nutrition.py`: recipe decomposition và macro.
- `configs/food_catalog.json`: ingredient/recipe knowledge base.
- `src/calcucalo/analyzer.py`: ghép toàn bộ pipeline.
- `src/calcucalo/api.py`: API cho mobile/backend.

## 4. JSON hiện hỗ trợ

### Contract gọn cho mobile

```json
{
  "food_id": "VN_COM_TAM",
  "name": "Cơm tấm sườn bì chả",
  "base_portion_g": 380,
  "components": [
    {
      "name": "Cơm tấm",
      "default_g": 200,
      "cal_per_100g": 130,
      "protein": 2.7,
      "fat": 0.3,
      "carb": 28.2
    },
    {
      "name": "Sườn nướng",
      "default_g": 100,
      "cal_per_100g": 240,
      "protein": 20.0,
      "fat": 17.0,
      "carb": 1.0
    },
    {
      "name": "Chả trứng",
      "default_g": 50,
      "cal_per_100g": 160,
      "protein": 11.0,
      "fat": 11.0,
      "carb": 4.0
    }
  ]
}
```

CLI:

```powershell
calcucalo analyze image.jpg --model model.onnx --json-format nutrition
```

Thêm `--component-pass` để bật lượt detect thứ hai trong crop món; dùng `--component-model` khi đã có checkpoint component chuyên biệt.

API:

```text
POST /api/v1/food/analyze
image=@image.jpg
response_format=nutrition
```

### JSON full cho debug và active learning

`response_format=full` còn chứa:

- `bbox_xyxy`, `mask_polygons`, detection confidence.
- `weight_g`, khoảng khối lượng, phương pháp và giả định.
- `estimated_components`, `estimated_totals`.
- `analysis_basis`, `visual_components_matched`, `data_quality`.
- `component_of` để tránh tính một thành phần hai lần.
- `user_override` khi gram đến từ thanh chỉnh của người dùng; giá trị này ưu tiên hơn model và catalog.
- warnings khi ảnh thiếu tỷ lệ hoặc catalog thiếu dữ liệu.

## 5. Mức chính xác hiện tại

Chưa có đủ bằng chứng để công bố một con số accuracy end-to-end cho mã nguồn này:

1. Checkpoint ONNX công khai đang dùng để smoke test chỉ có 58 lớp, không phải checkpoint 68 lớp mới được train bởi dự án này.
2. Chưa tải và train toàn bộ VietFood67 trong workspace hiện tại.
3. Chưa có test set chứa khối lượng cân thật và component mask.
4. Giá trị trong `food_catalog.json` hiện là **seed estimate cho phát triển phần mềm**. Chúng chưa được chuyên gia dinh dưỡng duyệt và không nên dùng cho quyết định y tế.

Các unit test chứng minh contract và thuật toán chạy đúng theo thiết kế; chúng không chứng minh model nhận diện đúng ngoài đời.

## 6. Những điểm cần cải thiện và cách cải thiện

### P0 — Train checkpoint VietFood67 đủ 68 lớp

Mục tiêu: thay checkpoint demo 58 lớp.

1. Tải VietFood67 và chạy validator để tìm ảnh thiếu nhãn, class ID sai và tọa độ ngoài `[0,1]`.
2. Kiểm tra trùng ảnh giữa train/val/test bằng perceptual hash; ảnh gần trùng làm mAP bị ảo.
3. Train lần lượt YOLO11n và YOLO11s; chỉ dùng YOLO11m khi GPU và latency cho phép.
4. Báo cáo AP theo từng lớp, confusion matrix và lỗi theo góc chụp/quán ăn, không chỉ báo cáo mAP chung.
5. Export ONNX và so sánh output `.pt`/`.onnx` trên cùng bộ golden images.

Lệnh khởi đầu:

```powershell
calcucalo prepare-dataset data\raw\vietfood67
python scripts\train_detector.py --data configs\vietfood67.yaml --model yolo11s.pt --epochs 150 --device 0 --export-onnx
```

### P1 — Tạo dataset component-level

VietFood67 không đủ để biết phần nào của cơm tấm là cơm, sườn, bì hay chả. Cần bổ sung dataset riêng:

- Mỗi ảnh có `dish_label` và instance mask cho từng component nhìn thấy.
- Thành phần bị che có cờ `occluded=true`; không ép annotator vẽ phần không thấy.
- Nhãn ghi phương pháp nấu: luộc, nướng, chiên, xào; ghi da/mỡ/sốt riêng.
- Mỗi ảnh có recipe variant, quán/nguồn, camera, góc chụp và ánh sáng.
- Khởi đầu bằng 15 món phổ biến, khoảng 300–500 ảnh/món; ưu tiên cơm tấm, phở, bún bò, bún chả, bánh mì và bún đậu.

Có thể dùng SAM 2 tạo pseudo-mask từ box/điểm rồi cho người gán nhãn sửa lại. Không dùng pseudo-mask chưa kiểm tra làm ground truth cuối cùng.

### P2 — Thu ground truth khối lượng

Mỗi sample cần:

- Khối lượng từng component trước khi ăn bằng cân bếp.
- Ảnh top-down và ảnh nghiêng 45°.
- Marker ArUco/thẻ chuẩn hoặc kích thước đĩa/bát thật.
- Đường kính và chiều cao bát cho món nước.
- Khối lượng còn lại sau ăn nếu ứng dụng cần tính lượng thực sự đã ăn.

Sau đó train regressor theo component với feature: mask area metric, perimeter, depth statistics, class embedding, camera angle, vessel geometry và cooking method. So sánh với baseline area × thickness × density; chỉ giữ model phức tạp nếu giảm MAE trên test set độc lập.

### P3 — Segmentation chuyên biệt

Kiến trúc đề xuất:

1. Dish detector tìm món tổng.
2. Crop vùng món và chạy component segmentation model.
3. Dùng SAM 2 làm annotator/pseudo-label, không làm nguồn class name.
4. Train YOLO11-seg hoặc Mask2Former trên mask đã được người kiểm tra.
5. Với món lẫn mạnh như cơm chiên, dùng semantic segmentation theo nhóm nguyên liệu thay vì ép instance cho từng hạt/thành phần nhỏ.

### P4 — Depth và thể tích đúng metric

Monocular depth chỉ cho depth tương đối nếu không có hiệu chuẩn. Để chuyển sang centimet:

- Phương án tốt cho mobile: ARCore/ARKit depth hoặc LiDAR nếu thiết bị hỗ trợ.
- Phương án phổ thông: hai ảnh có hướng dẫn góc chụp + marker kích thước biết trước.
- Phương án rẻ nhất: top-down + đường kính đĩa + prior độ dày đã fit từ dữ liệu cân thật.
- Món nước phải dùng hình học bát và mực nước; không dùng diện tích mặt thoáng đơn lẻ.

### P5 — Chuẩn hóa nutrition knowledge base

`configs/food_catalog.json` hiện là seed. Quy trình production nên là:

1. Import nguyên liệu cơ sở từ **Bảng thành phần thực phẩm Việt Nam** của Viện Dinh dưỡng.
2. Lưu ID nguồn, phiên bản, edible portion và đơn vị cho từng ingredient.
3. Recipe phải có raw weight, cooked weight và hệ số yield/retention sau nấu.
4. Tách dầu hấp thụ, nước sốt, nước dùng và topping thành component riêng.
5. Mỗi thay đổi recipe tạo version; meal log cũ giữ version đã dùng lúc tính.
6. Chuyên gia dinh dưỡng review trước khi đổi `data_quality` thành `validated`.

Nguồn chính thức tham khảo: [Bảng thành phần thực phẩm Việt Nam – Viện Dinh dưỡng](https://chuyentrang.viendinhduong.vn/viewfilenew/vi/thu-vien-sach-chuyen-nganh/189/1.html).

### P6 — Calibration confidence và active learning

- Calibrate confidence bằng temperature scaling/isotonic regression trên validation set.
- Khi confidence thấp hoặc top-2 gần nhau, yêu cầu người dùng chọn món.
- Cho phép sửa component và gram ngay trên mask.
- Lưu ảnh, prediction, correction, model version và catalog version vào hàng chờ review.
- Sampling ưu tiên lỗi confidence cao nhưng người dùng sửa, lớp hiếm và môi trường mới.

## 7. Bộ chỉ số bắt buộc trước production

| Bài toán | Chỉ số |
|---|---|
| Dish detection | mAP50-95, precision, recall và AP từng lớp |
| Component detection | Precision/recall/F1 từng component |
| Segmentation | Mask mAP50-95 và IoU/Dice |
| Khối lượng | MAE gram, median absolute error, MAPE theo món |
| Calories | MAE kcal và phần trăm bữa nằm trong ±10%, ±20% |
| Macro | MAE protein/fat/carb theo bữa |
| Confidence | ECE, reliability diagram, coverage-error curve |
| Runtime | p50/p95 latency, RAM/VRAM, model size trên thiết bị mục tiêu |

Test set phải tách theo **quán/người chuẩn bị và thời điểm**, không chỉ random theo ảnh. Nếu cùng một đĩa hoặc chuỗi ảnh gần nhau xuất hiện ở cả train và test, kết quả sẽ quá lạc quan.

Công cụ đã có sẵn:

```powershell
python scripts\validate_catalog.py --output outputs\catalog_validation.json
python scripts\evaluate_pipeline.py data\eval\manifest.jsonl `
  --model runs\detect\vietfood67_yolo11s\weights\best.pt `
  --component-pass `
  --output outputs\evaluation.json
```

Schema của từng dòng ground truth nằm tại `schemas/evaluation_sample.schema.json`.

## 8. Thứ tự triển khai khuyến nghị

1. Train và đánh giá detector VietFood67 đủ 68 lớp.
2. Chọn 6 món MVP, thu ảnh có cân thật và component mask.
3. Thẩm định ingredient database và recipe với chuyên gia dinh dưỡng.
4. Train component segmentation cho 6 món.
5. Fit mass regressor và đo calorie MAE end-to-end.
6. Chạy pilot, thu correction, active learning.
7. Mở rộng dần sang 15 rồi 30 món; không mở đủ 67 món trước khi có ground truth chất lượng.

## 9. Giấy phép và quyền riêng tư

- VietFood67 ghi giấy phép CC BY-NC-SA 4.0 và hạn chế sử dụng thương mại; cần xác minh với tác giả trước khi thương mại hóa.
- Metadata checkpoint ONNX tham khảo ghi AGPL-3.0.
- Ảnh bữa ăn có thể chứa khuôn mặt, vị trí hoặc dữ liệu sức khỏe. Cần xóa EXIF, làm mờ người và có cơ chế consent/retention rõ ràng trước khi lưu để active learning.

Tham khảo: [VietFood67 trên Kaggle](https://www.kaggle.com/datasets/thomasnguyen6868/vietfood68), [FoodDetector của tác giả dataset](https://github.com/nvhnam/FoodDetector), [Ultralytics SAM 2](https://docs.ultralytics.com/models/sam-2/), [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/).
