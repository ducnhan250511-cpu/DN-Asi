# DN-AI

**Think Smarter. Solve Anything.**

**Người phát triển: Đức Nhân**

DN-AI là ứng dụng web responsive có thể mở trên Android, iPhone và máy tính. Website ưu tiên kho kiến thức, công cụ tính toán/thống kê và checkpoint DN-AI Micro được đóng gói trong dự án; không gọi API chatbot của OpenAI, Gemini, Claude hay DeepSeek.

## Trạng thái thực tế

- Website React + TanStack Start + Vite.
- Giao diện chat responsive, lịch sử chat lưu trên trình duyệt.
- API chat nội bộ tại `/api/chat`.
- Mô hình Transformer DN-AI Micro có checkpoint đi kèm.
- Hỗ trợ công cụ giải toán/ký hiệu và thống kê CSV trong phạm vi mã nguồn hiện có.
- Vision chưa được huấn luyện; không nhận diện ảnh.
- Upload hiện hỗ trợ TXT, MD, CSV, JSON tối đa 12 KB. PDF/DOCX/XLSX chưa được bật.
- Checkpoint hiện tại có validation perplexity khoảng 2.941 trên 4 mẫu kiểm định; chế độ sinh tự do được tắt để tránh câu trả lời vô nghĩa. Hệ thống chỉ trả lời từ kho kiến thức/công cụ nội bộ và sẽ báo rõ khi không đủ dữ liệu.
- Người phát triển: Đức Nhân.

## Cách mở DN-AI trên điện thoại Android (không cần cài môi trường lập trình)

Cách đơn giản nhất là triển khai website lên Internet bằng GitHub + Vercel. Sau đó chỉ cần Chrome trên Android.

### 1. Đưa mã nguồn lên GitHub

1. Mở https://github.com/ bằng Chrome và đăng nhập.
2. Có thể tiếp tục dùng repository DN-AI hiện tại (`ducnhan250511-cpu/DN-AI`); không cần tạo repository mới.
3. Giải nén gói mã nguồn DN-AI đã cập nhật trên điện thoại hoặc máy tính. Thay nội dung mã nguồn cũ bằng nội dung trong gói mới, giữ `package.json`, `package-lock.json`, `src/`, `ai/`, `public/`, `scripts/`, `server/` và `migrations/` ở thư mục gốc repository.
4. Commit các thay đổi lên nhánh `main` để Vercel nhận commit mới và chạy deployment.
5. Không tải chính file ZIP như một tệp duy nhất lên repository. Không đưa `.env`, mật khẩu, token hoặc thông tin database lên GitHub.

> Nếu GitHub trên điện thoại khó tải nhiều thư mục, dùng Termux + Git hoặc máy tính để đẩy repository. Không đưa `.env`, mật khẩu hoặc API key lên GitHub.

### 2. Deploy bằng Vercel

1. Mở https://vercel.com/ và đăng nhập bằng GitHub.
2. Chọn **Add New → Project**, sau đó import repository `dn-ai`.
3. Framework preset: **TanStack Start**. Nếu Vercel không tự nhận diện, kiểm tra lại framework preset thay vì chọn một preset khác theo phỏng đoán.
4. Build command: `npm run build`.
5. Output directory: để mặc định theo cấu hình framework/Nitro; không tự đặt `public`.
6. Install command: `npm install --no-audit --no-fund`.
7. Nhấn Deploy và chờ kết quả.

Nếu build báo thiếu biến môi trường database/auth, không tự điền giá trị giả. Cần cấu hình dịch vụ database/auth thật theo log triển khai hoặc tạm thời tắt phần tính năng phụ thuộc database/auth trước khi deploy.

### 3. Mở website trên Android

1. Sau khi Vercel báo deployment thành công, mở URL `*.vercel.app` bằng Chrome Android.
2. Chọn menu Chrome (⋮) → **Add to Home screen** hoặc **Install app** nếu tùy chọn xuất hiện.
3. DN-AI sẽ có biểu tượng trên màn hình chính và mở như một ứng dụng web.

## Chạy local (máy tính hoặc Termux)

Cần Node.js 22+ và Python 3.10+ nếu muốn huấn luyện mô hình.

```bash
npm install
npm run dev
```

Mở `http://localhost:8080`.

## Mô hình AI

DN-AI hiện ưu tiên truy xuất câu trả lời đã biên soạn (RAG/FAQ) và công cụ xác định. Checkpoint Micro vẫn được giữ trong mã nguồn để tiếp tục nghiên cứu, nhưng không được bật sinh tự do khi chưa vượt qua kiểm định. Muốn mở lại tính năng này, cần tăng dữ liệu huấn luyện có chất lượng, tách tập kiểm định đủ lớn và đánh giá đầu ra tiếng Việt trước khi triển khai.

Checkpoint hiện tại là `dnai-micro` (Transformer 4 lớp, 4 attention heads, hidden size 128, vocabulary 1.268 token). Nó rất nhỏ và được huấn luyện trên tập dữ liệu thử nghiệm nội bộ. Không nên gọi đây là mô hình đa năng chất lượng cao.

Huấn luyện lại trên máy có Python và NumPy:

```bash
python -m pip install numpy
python -m ai.dnai.train --steps 700
```

Việc huấn luyện tiêu tốn tài nguyên và không nên thực hiện trên điện thoại cấu hình thấp. Hãy huấn luyện trên máy tính phù hợp.

## Giới hạn và bảo mật

- Không có API key AI bên thứ ba.
- Không gửi dữ liệu cá nhân/nhạy cảm vào ứng dụng thử nghiệm.
- Lịch sử chat hiện lưu trong localStorage của trình duyệt trên thiết bị.
- Rate limit hiện ở mức tiến trình máy chủ; cần giải pháp chia sẻ trạng thái nếu scale nhiều instance.
- Không có hệ thống đăng nhập hoàn chỉnh được xác minh trong bản đóng gói này.
- Không chạy mã do người dùng gửi lên máy chủ.
