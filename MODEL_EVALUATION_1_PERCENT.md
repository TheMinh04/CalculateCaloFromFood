# Đánh giá checkpoint VietFood67 train trên 1% dữ liệu

Ngày đánh giá: **25/09/2026**

Checkpoint được kiểm tra:

- `best.pt`
- `last.pt`

## 1. Kết luận ngắn

Hai checkpoint chứng minh quy trình train YOLO của dự án **đã chạy được từ đầu đến cuối** và model có thể nhận đúng cấu hình **68 lớp** của VietFood67. Đây là kết quả tốt cho một lần **smoke test**.

Tuy nhiên, model hiện tại **chưa đủ chính xác để dùng trong ứng dụng**. Các metric được lưu trong checkpoint rất thấp: mAP50 khoảng **0,019%** và recall khoảng **0,026%**. Model gần như chưa học được khả năng nhận diện ổn định sau chỉ 1 epoch.

Hai file này chỉ là **model object detection**. Chúng chưa tự học segmentation, chiều sâu, thể tích, khối lượng gram hoặc calories. Các phần khẩu phần và dinh dưỡng hiện vẫn do pipeline khác trong dự án xử lý bằng mask dự phòng, calibration, prior và catalog.

## 2. Phạm vi và cách đánh giá

Báo cáo được tạo bằng cách:

- Đọc metadata, cấu hình train và metric được lưu trong hai checkpoint.
- Nạp checkpoint bằng Ultralytics trong môi trường hiện tại.
- So sánh toàn bộ tensor trọng số của `best.pt` và `last.pt`.
- Đối chiếu tên lớp với `configs/vietfood67.yaml`.
- Kiểm tra số lượng file và phân bố nhãn của phần 1% dữ liệu mà Ultralytics lấy theo thứ tự tên file.

Không train lại model và chưa chạy lại một vòng benchmark độc lập trên toàn bộ tập test. Vì vậy, các metric dưới đây là metric được ghi trong checkpoint, không phải kết quả benchmark mới.

## 3. Cấu hình thực tế trong checkpoint

| Thuộc tính | Giá trị |
|---|---:|
| Kiến trúc | YOLO11n detection |
| Số tham số | 2.607.472 |
| Số lớp | 68 |
| Epoch | 1 |
| Tỷ lệ dữ liệu train | 1% |
| Kích thước ảnh | 320 px |
| Batch size | 32 |
| Pretrained | Có |
| Thiết bị lúc train | CUDA device 0 |
| Thời gian được ghi | 80,51 giây |
| Validation trong lệnh train | `val=False` |
| Warm-up cấu hình | 3 epoch |
| Phiên bản Ultralytics ghi trong checkpoint | 8.4.161 |

Dataset đang có trên máy:

| Split | Số ảnh | Số file nhãn |
|---|---:|---:|
| Train | 161.733 | 161.733 |
| Validation | 6.602 | 6.602 |
| Test | 3.284 | 3.284 |

Với `fraction=0.01`, Ultralytics lấy khoảng **1.617 ảnh train** đầu tiên sau khi sắp xếp đường dẫn. Phần dữ liệu này có 9.509 bounding box và xuất hiện đủ 68 class ID, nhưng phân bố rất lệch. Ví dụ:

- `Banh beo` và `Cao lau`: chỉ 1 box mỗi lớp.
- `Chao long`: 5 box.
- `Com chien duong chau`: 7 box.
- `Mi Quang`: 10 box.
- `Pho mai`: 3.169 box.

Như vậy, “đủ 68 lớp” không đồng nghĩa mỗi lớp đã có đủ mẫu để model học.

## 4. Metric hiện tại

| Metric trong checkpoint | Giá trị | Quy đổi | Đánh giá |
|---|---:|---:|---|
| Precision | 0,01176 | 1,176% | Rất thấp |
| Recall | 0,00026 | 0,026% | Gần như không phát hiện được đối tượng |
| mAP50 | 0,00019 | 0,019% | Chưa sử dụng được |
| mAP50-95 | 0,00019 | 0,019% | Chưa sử dụng được |
| Fitness | 0,00019 | 0,019% | Rất thấp |
| Validation box loss | 1,20570 | — | Chưa đủ epoch để kết luận xu hướng |
| Validation class loss | 4,89322 | — | Khả năng phân lớp còn rất yếu |
| Validation DFL loss | 1,23550 | — | Bounding box chưa hội tụ |

Không nên dùng các loss của một epoch duy nhất để kết luận model đang overfit hay underfit. Tuy nhiên, precision, recall và mAP cho thấy checkpoint này chưa đạt mức tối thiểu để thử nghiệm sản phẩm.

## 5. So sánh `best.pt` và `last.pt`

| Thuộc tính | `best.pt` | `last.pt` |
|---|---|---|
| Kích thước | 5.458.522 byte | 5.458.522 byte |
| SHA-256 | `DB0BA0C491148E9B05FD2A69AFCEB45D20732063359877D48E291A5206466AD4` | `30D570227739590A9DA076BE64128CEA1F7316B8C7947A3F7663354D54F4BE1D` |
| Số tensor khác nhau | 0/499 | 0/499 |
| Metric | Giống nhau | Giống nhau |

Hai file khác nhau về dữ liệu đóng gói nhưng **toàn bộ trọng số model giống nhau**. Điều này hợp lý vì quá trình chỉ chạy 1 epoch. Hiện tại có thể dùng `best.pt` theo quy ước, nhưng chất lượng suy luận của hai file là như nhau.

Checkpoint đã được strip optimizer: `epoch=-1`, không có optimizer, scaler hoặc EMA state. Có thể dùng nó làm trọng số để fine-tune, nhưng không nên xem đây là checkpoint đầy đủ để resume chính xác trạng thái optimizer của lần train cũ.

## 6. Những gì đã làm tốt

### 6.1. Quy trình train đã hoạt động

- Dataset YAML được YOLO đọc thành công.
- Model pretrained đã được chuyển sang head 68 lớp.
- Train hoàn tất và sinh được `best.pt`, `last.pt`.
- Cả hai file đều nạp được trong môi trường hiện tại.
- Tên và thứ tự 68 lớp khớp cấu hình VietFood67.

### 6.2. Model nhỏ, phù hợp làm baseline

YOLO11n chỉ có khoảng 2,61 triệu tham số và checkpoint khoảng 5,46 MB. Đây là lựa chọn hợp lý để kiểm thử nhanh, triển khai thử trên CPU/mobile và xây dựng pipeline trước khi tăng kích thước model.

### 6.3. Smoke test đã tìm ra vấn đề sớm

Lần chạy 1% cho thấy code, dataset và GPU có thể kết nối với nhau mà không cần tốn thời gian train toàn bộ dữ liệu. Đây chính là mục tiêu đúng của smoke test.

## 7. Những gì chưa tốt và nguyên nhân

### 7.1. Chỉ 1 epoch nên model chưa kịp học

Warm-up được đặt là 3 epoch nhưng toàn bộ quá trình chỉ chạy 1 epoch. Learning rate chưa đi hết giai đoạn warm-up, nên lần train này không thể đại diện cho năng lực thật của YOLO11n trên VietFood67.

### 7.2. Lấy 1% theo thứ tự file, không phải lấy mẫu cân bằng

Ultralytics lấy phần đầu của danh sách ảnh đã sắp xếp, không đảm bảo stratified sampling theo lớp. Một số lớp chỉ có 1–10 box trong tập train 1%, trong khi một lớp có hàng nghìn box. Model vì vậy bị thiếu dữ liệu cho nhiều lớp và dễ thiên lệch về lớp phổ biến.

### 7.3. Ảnh 320 px làm mất chi tiết nhỏ

Kích thước 320 phù hợp để smoke test tốc độ nhưng không lý tưởng cho các thành phần nhỏ như rau, trứng, chả, đồ ăn kèm hoặc topping. Training chính nên bắt đầu ở 640 px và chỉ giảm kích thước sau khi đã đo được ảnh hưởng đến độ chính xác.

### 7.4. Taxonomy đang trộn nhiều loại đối tượng

68 lớp hiện trộn:

- Món hoàn chỉnh: `Pho`, `Bun bo Hue`, `Com tam`.
- Thành phần: `Trung`, `Tom`, `Dau hu`, `Rau`.
- Đối tượng không phải thực phẩm: `Con nguoi`.

Điều này có thể hữu ích cho detection tổng quát nhưng chưa đủ để khẳng định model có thể bóc tách đầy đủ nguyên liệu trong một món. Dataset cần quy định rõ lớp nào là `dish`, lớp nào là `component` và component nào bắt buộc phải được gán nhãn.

### 7.5. Chưa đánh giá độc lập trên tập test

Chưa có báo cáo per-class AP, confusion matrix, false positive/false negative và kết quả trên ảnh điện thoại ngoài dataset. Metric trung bình hiện tại không cho biết lớp nào đang thất bại nhiều nhất.

### 7.6. Checkpoint không giải quyết khối lượng và dinh dưỡng

Bounding box không cung cấp chiều sâu hay khối lượng. Ngay cả khi detector nhận diện đúng món, model vẫn cần:

- Segmentation mask của từng thành phần.
- Vật chuẩn hoặc thông tin camera để suy ra kích thước thật.
- Depth/ảnh đa góc để ước lượng thể tích.
- Ground truth gram để hiệu chỉnh portion estimator.
- Nutrition catalog đã kiểm định.

## 8. Kế hoạch cải thiện đề xuất

### P0 — Train baseline thật trên nhiều dữ liệu hơn

Không tiếp tục đánh giá chất lượng từ checkpoint 1 epoch. Tạo một run mới từ `yolo11n.pt`:

- Dùng toàn bộ dữ liệu hoặc một subset được lấy mẫu cân bằng.
- Train 50–100 epoch, bật validation, dùng early stopping.
- Dùng `imgsz=640`.
- Lưu `results.csv`, confusion matrix và prediction samples.
- Giữ test split hoàn toàn tách biệt cho lần đánh giá cuối.

Lệnh baseline đề xuất trên Kaggle:

```bash
!python scripts/train_detector.py \
  --data "/kaggle/working/vietfood67.yaml" \
  --model "yolo11n.pt" \
  --epochs 50 \
  --image-size 640 \
  --batch 32 \
  --device "0" \
  --workers 4 \
  --patience 15 \
  --project "/kaggle/working/runs/detect" \
  --name "vietfood67_yolo11n_full_v1" \
  --export-onnx
```

Nếu Kaggle hết thời gian, cần lưu cả thư mục run/checkpoint sang Kaggle Output hoặc Dataset trước khi session kết thúc. Không nên dựa vào hai file hiện tại để resume optimizer vì optimizer state đã bị loại bỏ.

### P1 — Kiểm tra và cân bằng dữ liệu

- Thống kê số ảnh, số box và AP theo từng lớp.
- Tạo subset thử nghiệm theo stratified sampling thay vì `fraction=0.01` trực tiếp.
- Bổ sung hoặc oversample các lớp rất hiếm.
- Kiểm tra nhãn sai, box quá nhỏ, box trùng và ảnh rò rỉ giữa train/val/test.
- Quyết định có giữ lớp `Con nguoi` trong model thực phẩm hay không.

### P2 — Đánh giá detector đúng cách

Sau mỗi run chính, cần báo cáo tối thiểu:

- Precision, recall, mAP50 và mAP50-95 trên validation và test.
- AP của từng lớp.
- Confusion matrix.
- Kết quả theo kích thước đối tượng nhỏ/vừa/lớn.
- Tốc độ trên GPU, CPU và thiết bị triển khai mục tiêu.
- Ảnh false positive và false negative để sửa dataset.

Mục tiêu ban đầu có thể đặt là mAP50 từ 0,70, mAP50-95 từ 0,45 và recall từ 0,70 trên test; đây là ngưỡng định hướng, cần điều chỉnh theo yêu cầu sản phẩm và độ khó thực tế của dataset.

### P3 — Chỉ tăng kích thước model sau khi baseline sạch

Nếu dữ liệu đã được kiểm tra mà YOLO11n vẫn không đạt yêu cầu, thử YOLO11s hoặc YOLO11m. Không nên tăng model trước khi xử lý mất cân bằng và chất lượng nhãn vì model lớn không sửa được dữ liệu sai.

### P4 — Tách bài toán món ăn và thành phần

Để phân tích một món có nhiều thành phần, nên xây dựng dataset component-level riêng hoặc pipeline hai tầng:

1. Detector nhận diện món tổng.
2. Segmentation/detector thứ hai tìm từng thành phần trong vùng món.
3. Portion estimator tính gram của từng component.
4. Nutrition engine tính calories và macro từ gram.

VietFood67 hiện chủ yếu giúp bước nhận diện đối tượng; nó chưa cung cấp đầy đủ ground truth cho ba bước sau.

### P5 — Thu dữ liệu khối lượng và chiều sâu

- Chụp món từ góc chuẩn và có vật chuẩn kích thước biết trước.
- Cân riêng từng component để có gram ground truth.
- Thu ảnh đa góc hoặc depth sensor nếu có.
- Đo MAE/MAPE gram riêng, không suy diễn từ mAP detection.
- Đo MAE/MAPE calories trên món thật sau khi portion estimator ổn định.

## 9. Việc nên làm ngay

1. Giữ hai checkpoint hiện tại làm bằng chứng smoke test, không xem là model production.
2. Chạy một baseline YOLO11n 640 px với validation và tối thiểu 30–50 epoch.
3. Đánh giá `best.pt` mới trên toàn bộ test split.
4. Xem confusion matrix và AP từng lớp trước khi đổi kiến trúc.
5. Song song chuẩn bị annotation component mask và gram ground truth cho bài toán dinh dưỡng.

## 10. Kết luận cuối

Checkpoint 1% hiện tại đã đạt mục tiêu **kiểm tra quy trình**, chưa đạt mục tiêu **nhận diện chính xác**. Bước đúng tiếp theo không phải là tinh chỉnh ngưỡng confidence hay xuất ONNX, mà là train một baseline đủ epoch trên dữ liệu cân bằng, đánh giá độc lập trên test, rồi mới phát triển segmentation, depth và ước lượng gram.
