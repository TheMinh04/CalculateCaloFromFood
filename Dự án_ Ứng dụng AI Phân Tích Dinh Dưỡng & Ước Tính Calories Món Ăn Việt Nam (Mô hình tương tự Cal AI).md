**Dự án: Ứng dụng AI Phân Tích Dinh Dưỡng & Ước Tính Calories Món Ăn Việt Nam (Mô hình tương tự Cal AI)**

&nbsp;

# **I. GIỚI THIỆU**

## **1\. Mục tiêu dự án**

Xây dựng giải pháp di động thông minh giúp người dùng theo dõi sức khỏe và quản lý khẩu phần ăn cá nhân hóa bằng cách chụp ảnh bữa ăn. Ứng dụng tập trung giải quyết bài toán đặc thù của ẩm thực Việt Nam (món ăn phức hợp, đa thành phần, nhiều loại gia vị và món dạng nước).

&nbsp;

## **2\. Công nghệ cốt lõi**

**Computer Vision (YOLOv8-seg / YOLOv11-seg):** Phân đoạn thể hiện (Instance Segmentation) để bóc tách chính xác từng vùng thực phẩm riêng biệt (cơm, thịt, rau, trứng...) và xác định diện tích bề mặt.

&nbsp;

**Volume & Mass Estimation:** Kết hợp bản đồ độ sâu (Depth Estimation) và khối lượng riêng tiêu chuẩn (g/cm^3) để ước lượng khối lượng (g).

&nbsp;

**Multimodal LLM (Kiến trúc Hybrid):** Đóng vai trò chuyên gia ẩm thực tinh chỉnh các yếu tố khó thấy qua ảnh 2D (độ ngấm dầu mỡ, nước sốt, thành phần chìm dưới nước dùng).

&nbsp;

**Knowledge Base chuẩn hóa:** Tích hợp trực tiếp Bảng thành phần thực phẩm của Viện Dinh dưỡng Quốc gia để tra cứu chính xác Calories, Protein, Carb, Fat và Chất xơ.

&nbsp;

## **3\. Luồng trải nghiệm người dùng (User Flow)**

**Chụp ảnh:** Người dùng chụp đĩa thức ăn qua camera ứng dụng.

&nbsp;

**Phân tích tự động:** Hệ thống quét, bóc tách các món ăn và tính toán macro trong vài giây.

&nbsp;

**Hiệu chỉnh nhanh:** Người dùng xem lại mask phân vùng của AI, điều chỉnh thanh trượt khối lượng hoặc tick chọn nhanh các tùy chọn (bỏ mỡ, không húp nước dùng).

&nbsp;

**Theo dõi & Thống kê:** Dữ liệu tự động đồng bộ vào nhật ký dinh dưỡng, so sánh với chỉ số TDEE/mục tiêu thể trạng (giảm cân, giữ dáng, tăng cơ).

# **II. CHỨC NĂNG CHÍNH**

## **1\. Phân hệ Thị giác máy tính & Phân tích Dinh dưỡng (Core AI Engine)**

Đây là khối tính năng nền tảng chịu trách nhiệm xử lý ảnh, bóc tách nguyên liệu và tính toán macro.

**1.1. Tiếp nhận và tiền xử lý hình ảnh (Image Ingestion & Pre-processing):**

Nhận ảnh từ mobile client (dạng multipart/form-data hoặc base64).

Chuẩn hóa kích thước (resize về 640x640 hoặc 1024x1024), xoay đúng hướng ảnh (EXIF orientation).

Nén ảnh tối ưu dung lượng trước khi nạp vào pipeline suy luận để giảm latency mạng.

**1.2. Phân đoạn thực phẩm (Food Instance Segmentation):**

Tích hợp model YOLOv8-seg (hoặc YOLOv11-seg) đã fine-tune trên tập dữ liệu món ăn Việt Nam.

Trích xuất tọa độ Bounding Box, ma trận điểm ảnh (Polygon Mask) và nhãn phân loại (Class ID) cho từng thành phần (cơm, thịt kho, trứng rán, canh...).

Tính toán diện tích bề mặt theo tỷ lệ phần trăm khung hình ($Pixel Area$).

**1.3. Ước lượng khối lượng theo thể tích (Portion/Volume Estimation):**

Kết hợp thuật toán nội suy hoặc Monocular Depth Model (như Depth Anything) để đo độ sâu/độ gồ ghề của món ăn.

Nhân diện tích pixel với độ sâu trung bình để quy ra thể tích ước lượng ($cm^3$).

Nội suy khối lượng (g) \= Thể tích (cm^3) x Khối lượng riêng chuẩn của nguyên liệu (g/cm^3).

**1.4. Bộ tinh chỉnh đa phương thức (Vision LLM Post-Processing):**

Gửi ảnh gốc kèm kết quả sơ bộ từ YOLO sang Vision LLM (Gemini 2.0 Flash hoặc GPT-4o mini) qua API.

Prompt chuyên sâu về văn hóa ẩm thực Việt để xử lý các chi tiết YOLO không nhìn thấy: độ ngấm dầu mỡ (món xào/chiên), nước sốt chan trên bề mặt, topping chìm trong nước dùng (bún, phở).

Trả về dữ liệu JSON có cấu trúc (Structured Outputs): danh sách món, khối lượng thực ước tính (g), độ tin cậy.

**1.5. Tra cứu dinh dưỡng (Nutrition Engine):**

Map trực tiếp mã món với Bảng thành phần thực phẩm Việt Nam (Viện Dinh dưỡng Quốc gia) lưu trong database.

Tính toán các chỉ số:

Calories (kcal)

Protein (g)

Lipid / Fat (g)

Carbohydrate (g)

Chất xơ / Fiber (g)

## **2\. Phân hệ Ứng dụng Di động & Giao diện Người dùng (Mobile App UI/UX)**

**2.1. Module Camera chụp ảnh bữa ăn:**

Khung viewfinder thông minh hướng dẫn người dùng: nhắc nhở góc chụp nghiêng 45° hoặc chụp thẳng từ trên xuống để thấy toàn bộ đĩa/bát.

Tùy chọn tải ảnh có sẵn từ thư viện máy hoặc chụp trực tiếp.

**2.2. Màn hình Xem trước & Hiệu chỉnh kết quả (Scan Result & Adjustment):**

Hiển thị ảnh chụp có vẽ đè bounding box/polygon mask bao quanh từng món ăn đã nhận diện.

Thanh trượt điều chỉnh khẩu phần (Serving Slider): Người dùng có thể kéo tăng/giảm khối lượng nếu ăn nhiều hoặc ít hơn mức ước lượng chuẩn (ví dụ: kéo từ 150g cơm lên 200g).

Tùy chọn chi tiết: Tick chọn các option đặc thù món Việt: *"Không chan nước béo"*, *"Bỏ da gà"*, *"Ít mỡ hành"*, *"Nước dùng không húp hết"*.

Nút bổ sung thủ công nếu AI bỏ sót món hoặc cho phép chọn đổi món tương đương từ thanh tìm kiếm.

**2.3. Nhật ký dinh dưỡng theo ngày (Daily Food Diary):**

Phân loại bữa ăn: Bữa sáng, Bữa trưa, Bữa tối, Bữa phụ/Snack.

Hiển thị tổng lượng Calo nạp vào so với chỉ số TDEE mục tiêu (Calories In vs. Calories Target).

Biểu đồ tròn phân bổ tỷ lệ 3 chất đa lượng chính: Carb \- Protein \- Fat.

## **3\. Phân hệ Hồ sơ Người dùng & Tính toán Cá nhân hóa (User Profile & Health Tracking)**

**3.1. Quản lý tài khoản:**

Đăng ký/Đăng nhập qua Email, Google, Apple ID.

Quản lý thông tin cá nhân: Giới tính, Tuổi, Chiều cao (cm), Cân nặng hiện tại (kg), Cân nặng mục tiêu (kg).

**3.2. Đánh giá chỉ số cơ thể & Thiết lập mục tiêu:**

Tính toán tự động chỉ số khối cơ thể (BMI).

Tính toán tốc độ chuyển hóa cơ bản (BMR) và tổng mức tiêu hao năng lượng hàng ngày (TDEE) theo mức độ vận động (ít vận động, vừa phải, vận động nặng).

Thiết lập mục tiêu: Giảm cân (Thâm hụt calo 300 \- 500kcal/ngày), Tăng cơ/Tăng cân (Thặng dư calo), hoặc Duy trì vóc dáng.

**3.3. Báo cáo & Thống kê chu kỳ (Analytics & Insights):**

Biểu đồ đường theo dõi cân nặng biến thiên theo tuần/tháng.

Biểu đồ cột theo dõi mức độ tuân thủ hạn mức calo từng ngày.

Đưa ra cảnh báo nếu bữa ăn thiếu hụt protein hoặc vượt quá ngưỡng sodium/chất béo cho phép.

## **4\. Phân hệ Backend, API & Cơ sở dữ liệu (Server Architecture)**

**4.1. Dịch vụ AI Inference (AI Microservice):**

Xây dựng bằng Python (FastAPI/TorchServe) để load file mô hình YOLO (.pt hoặc .onnx).

Cung cấp endpoint: POST /api/v1/food/detect (nhận image file $\\rightarrow$ trả về bounding boxes, masks, predicted classes, estimated weight).

Tích hợp cơ chế Queue (Celery \+ Redis) nếu lượng request đồng thời cao, tránh nghẽn GPU/CPU server.

**4.2. Dịch vụ Quản lý Dữ liệu & Nghiệp vụ (Main Backend Service):**

Xử lý Auth (JWT Tokens), đồng bộ lịch sử ăn uống, quản lý tiến độ người dùng.

Tích hợp Cloud Storage (AWS S3 hoặc Cloudinary) để lưu trữ ảnh chụp người dùng tải lên.

**4.3. Thiết kế Cơ sở dữ liệu (PostgreSQL Database):**

users: Thông tin định danh, cân nặng, chiều cao, TDEE target.

foods: Bảng thành phần chuẩn Viện Dinh Dưỡng (id, name, calories, protein, fat, carb, density g cm3).

meals: Bản ghi bữa ăn (id, user\_id, meal\_type, image\_url, created\_at).

meal\_items: Chi tiết từng món trong bữa ăn (meal\_id, food\_id, estimated\_weight\_g, adjusted\_weight\_g, calories, macros).

## **5\. Phân hệ Quản trị Dữ liệu (Admin & Active Learning Dashboard)**

**5.1. Quản lý kho thực phẩm (Food Metadata CMS):**

CRUD thông tin món ăn, cập nhật hệ số calo, khối lượng chuẩn cho từng đĩa/bát.

**5.2. Thu thập dữ liệu phản hồi (Active Learning Pipeline):**

Lưu lại các trường hợp người dùng chỉnh sửa nhãn món ăn (khi AI đoán sai) để đưa vào tập dữ liệu chờ gán nhãn.

Giúp đội ngũ phát triển liên tục mở rộng và tái huấn luyện (re-train) mô hình YOLO theo thời gian thực mà không cần đi chụp ảnh thủ công từ đầu.

&nbsp;