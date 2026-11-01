# Hướng dẫn chạy DN-AI trên Android

**Người phát triển: Đức Nhân**

## Khuyến nghị: dùng website đã deploy

Bạn không cần cài Python hay Node.js trên điện thoại để sử dụng website. Dùng Chrome Android:

1. Giải nén `DN-AI.zip` bằng ứng dụng quản lý tệp.
2. Tạo repository GitHub mới và tải nội dung đã giải nén lên (các thư mục phải ở cấp gốc).
3. Import repository đó vào Vercel.
4. Đặt Build Command là `npm run build`, Install Command là `npm install --no-audit --no-fund`.
5. Deploy. Mở URL Vercel trên Chrome.
6. Trong Chrome chọn ⋮ → Add to Home screen / Install app.

Nếu Vercel build thất bại, mở mục Build Logs và xử lý đúng lỗi được báo. Không thêm API key giả hoặc mật khẩu vào mã nguồn.

## Lưu ý

DN-AI Micro là mô hình ngôn ngữ thử nghiệm quy mô nhỏ. Bản hiện tại ưu tiên kho kiến thức và công cụ nội bộ; chế độ sinh tự do đang tắt vì chất lượng kiểm định chưa đạt yêu cầu. Website không gọi API chatbot thương mại. Chưa có nhận diện ảnh; hiện chỉ đọc tệp TXT, MD, CSV, JSON dung lượng tối đa 12 KB.
