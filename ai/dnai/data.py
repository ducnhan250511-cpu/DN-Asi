"""Original DN-AI seed dataset.

Written for this project. No scraped copyrighted books and no personal data.
Cleaning: NFC, whitespace collapse, exact dedup.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

ARTICLES: list[dict] = [
    {
        "id": "ai-basics",
        "body": "Trí tuệ nhân tạo (AI) là lĩnh vực tạo ra hệ thống máy tính có thể thực hiện những nhiệm vụ thường cần năng lực con người như nhận dạng mẫu, hiểu ngôn ngữ, dự đoán và hỗ trợ ra quyết định. AI không phải con người và không mặc nhiên hiểu đúng mọi thứ; kết quả phụ thuộc dữ liệu, cách thiết kế và kiểm tra.",
        "qa": [
            ("Trí tuệ nhân tạo là gì? Có ba ứng dụng thực tế nào?", "Trí tuệ nhân tạo (AI) là công nghệ giúp máy tính xử lý dữ liệu để thực hiện một số nhiệm vụ như nhận dạng, dự đoán và hiểu ngôn ngữ. Ba ứng dụng gần gũi là: (1) bản đồ dự đoán tình trạng giao thông và gợi ý đường đi; (2) dịch máy và chuyển giọng nói thành văn bản; (3) lọc thư rác và gợi ý nội dung phù hợp. AI vẫn có thể sai nên cần kiểm tra kết quả."),
            ("AI là gì?", "AI (trí tuệ nhân tạo) là lĩnh vực tạo ra hệ thống máy tính có thể thực hiện một số nhiệm vụ thường cần trí tuệ con người, chẳng hạn nhận dạng hình ảnh, xử lý ngôn ngữ và dự đoán. AI không phải con người và có thể mắc lỗi."),
            ("Cho ví dụ ứng dụng trí tuệ nhân tạo trong đời sống", "Ba ví dụ: bản đồ dự đoán giao thông và gợi ý đường đi; ứng dụng dịch ngôn ngữ hoặc chuyển giọng nói thành văn bản; hệ thống lọc thư rác và gợi ý nội dung. Các hệ thống này hỗ trợ con người, nhưng kết quả vẫn cần được kiểm tra."),
        ],
    },
    {
        "id": "creator",
        "body": "DN-AI là dự án do Đức Nhân phát triển. Tên người phát triển/đơn vị thực hiện được ghi là Đức Nhân.",
        "qa": [
            ("Ai là người phát triển DN-AI?", "DN-AI được phát triển bởi Đức Nhân."),
            ("Ai làm ra DN-AI?", "Người phát triển DN-AI: Đức Nhân."),
            ("Công ty hoặc người tạo ra DN-AI là ai?", "Thông tin người phát triển dự án: Đức Nhân."),
        ],
    },
    {
        "id": "identity",
        "body": "DN-AI là nền tảng trí tuệ nhân tạo tự phát triển. Slogan là Think Smarter. Solve Anything. Mô hình ngôn ngữ DN-AI Micro là Transformer do dự án tự huấn luyện, không gọi OpenAI, Gemini, Claude hay API chatbot khác.",
        "qa": [
            ("DN-AI là gì?", "DN-AI là nền tảng AI tự phát triển, với slogan Think Smarter. Solve Anything. Mình giúp chat, học, viết, đọc tài liệu và phân tích dữ liệu bằng mô hình riêng của dự án."),
            ("Bạn có dùng ChatGPT hay OpenAI không?", "Không. Mình là DN-AI Micro, Transformer do dự án tự huấn luyện. Mình không gọi OpenAI, Gemini, Claude hay API chatbot thương mại nào."),
            ("What is DN-AI?", "DN-AI is a self-developed AI platform. The slogan is Think Smarter. Solve Anything. The Micro model is a Transformer trained by this project, with no third-party chatbot API."),
            ("Xin chào", "Chào bạn! Mình là DN-AI. Bạn muốn hỏi kiến thức, tính toán hay tra cứu một tài liệu?"),
            ("Bạn tên gì?", "Mình là DN-AI, trợ lý AI do Đức Nhân phát triển."),
            ("Cảm ơn nhé", "Không có gì! Khi cần, bạn cứ gửi câu hỏi tiếp theo nhé."),
        ],
    },
    {
        "id": "transformer",
        "body": "Transformer xử lý chuỗi bằng self-attention. Mỗi token được nhúng thành vector, cộng thông tin vị trí bằng RoPE, rồi đi qua nhiều lớp attention nhiều đầu và mạng feed-forward. Mô hình nhân quả chỉ nhìn token phía trước khi sinh chữ.",
        "qa": [
            ("Transformer hoạt động ra sao?", "Transformer đổi token thành vector, gắn vị trí bằng RoPE, rồi chồng các lớp self-attention nhiều đầu và feed-forward. Khi sinh chữ, mô hình nhân quả chỉ được nhìn các token đã có."),
            ("RoPE là gì?", "RoPE là rotary positional encoding. Nó xoay cặp chiều của query và key theo vị trí, để attention biết token nào đứng trước token nào mà không cần cộng một vector vị trí riêng."),
            ("What is self-attention?", "Self-attention lets each token score every earlier token, then mix their values. Multi-head attention does this in several subspaces at once."),
        ],
    },
    {
        "id": "sampling",
        "body": "Khi sinh văn bản, logits được chia cho temperature. Temperature thấp làm câu chắc và lặp khuôn hơn. Temperature cao làm câu đa dạng nhưng dễ lệch. Top-k chỉ giữ k token điểm cao nhất. Top-p giữ tập token có tổng xác suất vừa đạt p.",
        "qa": [
            ("Temperature trong sinh chữ là gì?", "Temperature chia logits trước softmax. Thấp thì mô hình bám token chắc nhất. Cao thì các token khác có cơ hội, câu đa dạng hơn nhưng dễ lan man."),
            ("Top-k và top-p khác nhau thế nào?", "Top-k cắt còn k token điểm cao nhất. Top-p cắt theo tổng xác suất: lấy ít token nếu một token đã rất chắc, lấy nhiều hơn nếu phân phối đều."),
        ],
    },
    {
        "id": "checkpoint",
        "body": "Checkpoint lưu trọng số, từ vựng và cấu hình. Huấn luyện có thể dừng rồi nạp lại để học tiếp. Website không đọc file trọng số thô từ trình duyệt. Inference chạy ở máy chủ mô hình hoặc trong tiến trình server của ứng dụng.",
        "qa": [
            ("Checkpoint dùng để làm gì?", "Checkpoint lưu trọng số, bộ từ vựng và cấu hình để nạp lại khi sinh chữ hoặc học tiếp. Thiếu checkpoint thì hệ thống phải báo chưa sẵn sàng, không được bịa câu trả lời."),
            ("Vercel có huấn luyện mô hình lớn không?", "Không. Vercel chạy website và API nhẹ. Huấn luyện và mô hình lớn cần máy có CPU hoặc GPU riêng, rồi nối bằng biến môi trường tới model server."),
        ],
    },
    {
        "id": "study",
        "body": "Ôn tập hiệu quả khi chia nhỏ, nhớ lại chủ động và giãn cách. Flashcard hỏi một ý, lật đáp án, rồi xếp thẻ dễ sang hộp dài ngày hơn. Kế hoạch học nên có mục tiêu từng buổi, bài luyện ngắn và phần tóm tắt cuối.",
        "qa": [
            ("Làm sao học thuộc bền hơn?", "Chia nhỏ bài, tự nhớ lại trước khi xem đáp án, và giãn cách ôn. Flashcard hợp việc này: một mặt câu hỏi, một mặt ý chính, thẻ dễ thì ôn thưa hơn."),
            ("Lập kế hoạch học một chương thế nào?", "Một buổi nên có mục tiêu hẹp, 25 đến 40 phút tập trung, vài câu luyện, rồi tóm tắt lại bằng lời của mình. Cuối tuần ôn các thẻ còn khó."),
            ("How do flashcards work?", "One side asks a single idea, the other holds the answer. Recall first, then flip. Easy cards wait longer before the next review."),
        ],
    },
    {
        "id": "writing",
        "body": "Một đoạn rõ có câu chủ đề, phần triển khai và câu chốt. Email ngắn cần lời chào, việc cần nói, thời hạn nếu có, và lời cảm ơn. Đổi phong cách là đổi từ ngữ và nhịp câu, không đổi sự kiện.",
        "qa": [
            ("Viết email xin gia hạn thì gồm gì?", "Mở đầu chào đúng người, nêu việc cần gia hạn, lý do ngắn, mốc thời gian mới, và cảm ơn. Giữ sự kiện đúng, không thêm chi tiết không có."),
            ("Tóm tắt một đoạn văn ra sao?", "Giữ ý chính và số liệu, bỏ ví dụ lặp. Một hoặc hai câu là đủ nếu đoạn gốc ngắn. Không bịa thêm ý không có trong nguồn."),
            ("How should a paragraph be built?", "Start with the point, follow with the reason or detail, and close the idea. Changing style means new wording, not new facts."),
        ],
    },
    {
        "id": "math-linear",
        "body": "Phương trình bậc nhất một ẩn có dạng ax + b = c với a khác 0. Trừ b ở hai vế rồi chia cho a. Nghiệm là x = (c - b) / a. Công cụ ký hiệu của DN-AI tính đúng từng bước; mô hình ngôn ngữ không được bịa số.",
        "qa": [
            ("Giải phương trình bậc nhất thế nào?", "Đưa về ax + b = c. Trừ b cả hai vế, rồi chia cho a nếu a khác 0. Nghiệm là x = (c - b) / a. Số cụ thể nên để công cụ ký hiệu tính."),
            ("Vì sao không để mô hình chữ đoán phép tính?", "Mô hình ngôn ngữ dự đoán token, không phải máy tính. Phép tính và phương trình cụ thể đi qua solver để ra số đúng, rồi mới giải thích."),
        ],
    },
    {
        "id": "quadratic",
        "body": "Phương trình bậc hai ax^2 + bx + c = 0. Delta bằng b^2 - 4ac. Delta âm thì vô nghiệm thực. Delta bằng 0 thì một nghiệm kép x = -b / (2a). Delta dương thì hai nghiệm x = (-b ± căn delta) / (2a).",
        "qa": [
            ("Công thức nghiệm bậc hai là gì?", "Với ax^2 + bx + c = 0, tính delta = b^2 - 4ac. Nếu delta dương có hai nghiệm (-b ± căn delta) / (2a). Delta bằng 0 thì nghiệm kép. Delta âm thì không có nghiệm thực."),
        ],
    },
    {
        "id": "percent",
        "body": "Phần trăm là phần trên 100. p phần trăm của N bằng N nhân p chia 100. Tăng thêm p phần trăm là nhân với (1 + p/100). Giảm p phần trăm là nhân với (1 - p/100).",
        "qa": [
            ("Cách tính phần trăm?", "p phần trăm của N là N * p / 100. Muốn tăng p phần trăm thì nhân N với 1 + p/100. Muốn giảm thì nhân với 1 - p/100."),
        ],
    },
    {
        "id": "circle",
        "body": "Hình tròn bán kính r có chu vi 2 * pi * r và diện tích pi * r^2. Đường kính gấp đôi bán kính. Pi xấp xỉ 3.1416 khi cần số thập phân.",
        "qa": [
            ("Công thức diện tích hình tròn?", "Diện tích hình tròn là pi nhân bình phương bán kính. Chu vi là 2 pi r. Đường kính bằng 2r."),
        ],
    },
    {
        "id": "pythagoras",
        "body": "Trong tam giác vuông, bình phương cạnh huyền bằng tổng bình phương hai cạnh góc vuông: a^2 + b^2 = c^2. Cạnh huyền là cạnh đối diện góc vuông.",
        "qa": [
            ("Định lý Pythagoras nói gì?", "Ở tam giác vuông, a^2 + b^2 = c^2, với c là cạnh huyền. Dùng để tìm cạnh còn lại khi biết hai cạnh kia."),
        ],
    },
    {
        "id": "stats",
        "body": "Trung bình cộng là tổng các số chia cho số lượng. Trung vị là giá trị đứng giữa sau khi sắp xếp. Độ lệch chuẩn đo độ phân tán quanh trung bình. Tương quan gần 1 hoặc -1 là hai cột đi cùng chiều hoặc ngược chiều, không tự chứng minh nhân quả.",
        "qa": [
            ("Trung bình và trung vị khác gì?", "Trung bình cộng lấy tổng chia cho số phần tử, dễ bị giá trị cực đoan kéo. Trung vị là mốc giữa của dãy đã sắp, vững hơn khi có số lệch."),
            ("Tương quan có phải là nguyên nhân không?", "Không. Tương quan chỉ nói hai cột cùng tăng hoặc ngược nhau. Kết luận nguyên nhân cần thí nghiệm hoặc lý do riêng, không chỉ một hệ số."),
        ],
    },
    {
        "id": "photosynthesis",
        "body": "Quang hợp là quá trình cây xanh dùng ánh sáng, nước và khí cacbonic để tạo glucose và nhả oxy. Diễn ra chủ yếu trong lục lạp nhờ diệp lục. Phương trình khái quát: nước + CO2 + ánh sáng cho ra glucose và oxy.",
        "qa": [
            ("Quang hợp là gì?", "Quang hợp là lúc cây xanh dùng ánh sáng, nước và CO2 để tạo glucose và thải oxy. Chỗ chính là lục lạp, nhờ diệp lục."),
            ("What is photosynthesis?", "Photosynthesis is how green plants use light, water and carbon dioxide to make glucose and release oxygen, mainly inside chloroplasts."),
        ],
    },
    {
        "id": "newton",
        "body": "Định luật I Newton: nếu tổng lực bằng không, vật đứng yên sẽ đứng yên, vật đang chuyển động thẳng đều sẽ giữ nguyên vận tốc. Quán tính là xu hướng giữ trạng thái chuyển động đó.",
        "qa": [
            ("Định luật I Newton nói gì?", "Khi tổng lực bằng không, vật giữ nguyên trạng thái: đứng yên hoặc chuyển động thẳng đều. Đó là quán tính."),
        ],
    },
    {
        "id": "boiling",
        "body": "Nước tinh khiết sôi ở khoảng 100 độ C tại áp suất một atmosphere. Lên núi, áp suất thấp hơn nên nước sôi ở nhiệt độ thấp hơn. Muối hòa tan làm nhiệt độ sôi tăng nhẹ.",
        "qa": [
            ("Nước sôi ở bao nhiêu độ?", "Ở áp suất một atmosphere, nước tinh khiết sôi khoảng 100 độ C. Áp suất thấp hơn thì nhiệt độ sôi giảm."),
        ],
    },
    {
        "id": "solar",
        "body": "Hệ Mặt Trời có Mặt Trời ở giữa và tám hành tinh: Thủy, Kim, Trái Đất, Hỏa, Mộc, Thổ, Thiên Vương, Hải Vương. Trái Đất là hành tinh thứ ba, có một vệ tinh tự nhiên là Mặt Trăng.",
        "qa": [
            ("Hệ Mặt Trời có những hành tinh nào?", "Tám hành tinh theo thứ tự: Thủy tinh, Kim tinh, Trái Đất, Hỏa tinh, Mộc tinh, Thổ tinh, Thiên Vương và Hải Vương. Trái Đất đứng thứ ba."),
            ("Name the planets in order.", "Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune. Earth is third and has one natural satellite, the Moon."),
        ],
    },
    {
        "id": "dna",
        "body": "DNA mang thông tin di truyền, gồm hai mạch xoắn kép. Bốn base là adenine, thymine, guanine và cytosine. A cặp với T, G cặp với C. Gene là đoạn DNA mã hóa một sản phẩm như protein.",
        "qa": [
            ("DNA là gì?", "DNA là phân tử mang thông tin di truyền, dạng xoắn kép. A cặp T, G cặp C. Gene là một đoạn DNA có chức năng."),
        ],
    },
    {
        "id": "vietnam",
        "body": "Việt Nam ở Đông Nam Á, thủ đô là Hà Nội, thành phố lớn nhất là Thành phố Hồ Chí Minh. Đất nước trải dài ven Biển Đông, phía bắc giáp Trung Quốc, phía tây giáp Lào và Campuchia.",
        "qa": [
            ("Thủ đô Việt Nam là gì?", "Thủ đô Việt Nam là Hà Nội. Thành phố lớn nhất theo dân số là Thành phố Hồ Chí Minh."),
            ("Việt Nam giáp những nước nào?", "Phía bắc giáp Trung Quốc, phía tây giáp Lào và Campuchia, phía đông và nam trông ra Biển Đông."),
        ],
    },
    {
        "id": "vietnamese",
        "body": "Chữ Quốc ngữ là chữ Latinh có dấu thanh: sắc, huyền, hỏi, ngã, nặng, và không dấu. Tokenizer của DN-AI chuẩn hóa Unicode NFC để các dấu không bị tách sai thành mã tổ hợp.",
        "qa": [
            ("Vì sao phải chuẩn hóa Unicode khi tách từ tiếng Việt?", "Một chữ có dấu có thể viết bằng ký tự dựng sẵn hoặc dấu rời. NFC gộp về một dạng để từ điển không bị vỡ đôi và dấu không rơi khỏi chữ."),
        ],
    },
    {
        "id": "variable",
        "body": "Biến trong lập trình là tên gắn với một giá trị trong bộ nhớ. Gán lại thì giá trị đổi, tên giữ nguyên. Hằng là tên không cho gán lại. Kiểu dữ liệu nói giá trị đó là số, chữ, đúng sai hay cấu trúc.",
        "qa": [
            ("Biến là gì trong lập trình?", "Biến là tên trỏ tới một giá trị. Bạn gán giá trị mới thì nội dung đổi. Kiểu dữ liệu cho biết đó là số, chuỗi, boolean hay cấu trúc khác."),
            ("What is a variable?", "A variable is a name bound to a value. Reassigning changes the value, not the name. A constant is a binding you do not reassign."),
        ],
    },
    {
        "id": "loop",
        "body": "Vòng lặp lặp một khối lệnh. for đi qua từng phần tử hoặc một dải số. while lặp khi điều kiện còn đúng. Vòng vô hạn xảy ra nếu điều kiện không bao giờ thành sai. Nên có điểm dừng rõ.",
        "qa": [
            ("for và while khác nhau ra sao?", "for hợp khi biết dãy cần đi qua. while hợp khi lặp cho tới một điều kiện. Cả hai đều cần cách thoát, nếu không chương trình không dừng."),
        ],
    },
    {
        "id": "function",
        "body": "Hàm là khối lệnh có tên, nhận đối số và có thể trả giá trị. Hàm giúp khỏi lặp code và dễ kiểm thử. Tham số là tên trong hàm, đối số là giá trị lúc gọi.",
        "qa": [
            ("Hàm dùng để làm gì?", "Hàm gom một việc vào một tên, nhận đối số và trả kết quả. Nhờ vậy không copy logic, và có thể kiểm thử riêng từng hàm."),
        ],
    },
    {
        "id": "debug",
        "body": "Debug là tìm vì sao chương trình chạy sai. Đọc thông báo lỗi từ trên xuống, thu hẹp chỗ nghi, in hoặc xem giá trị trung gian, rồi sửa một thứ mỗi lần. Đừng đổi nhiều chỗ cùng lúc.",
        "qa": [
            ("Các bước debug cơ bản?", "Đọc lỗi, xác định dòng liên quan, kiểm tra giá trị thật sự, đặt giả thuyết nhỏ, sửa một chỗ rồi chạy lại. Sửa nhiều thứ một lúc sẽ không biết cái nào đã chữa đúng."),
        ],
    },
    {
        "id": "html",
        "body": "HTML mô tả cấu trúc trang: tiêu đề, đoạn, liên kết, nút, ảnh. CSS trình bày. JavaScript xử lý hành vi. Nút bấm cần việc thật, không để trang trí trống.",
        "qa": [
            ("HTML, CSS và JavaScript mỗi thứ làm gì?", "HTML là cấu trúc, CSS là cách nhìn, JavaScript là hành vi. Một nút nên gắn việc cụ thể chứ không chỉ để có trên trang."),
        ],
    },
    {
        "id": "algorithm",
        "body": "Thuật toán là dãy bước hữu hạn để ra kết quả. Cần đúng, dừng được, và càng ít bước càng tốt khi dữ liệu lớn. Độ phức tạp thường viết theo n, ví dụ tuyến tính là đi một lần qua n phần tử.",
        "qa": [
            ("Thuật toán là gì?", "Thuật toán là các bước rõ ràng, làm xong trong số lần hữu hạn, và cho kết quả cần tìm. Khi n tăng, ta quan tâm số bước tăng nhanh đến mức nào."),
        ],
    },
    {
        "id": "git",
        "body": "Git lưu lịch sử thay đổi của mã nguồn. Commit là một mốc có thông điệp. Nhánh cho phép thử việc riêng rồi hợp lại. Không commit mật khẩu hay khóa API.",
        "qa": [
            ("Commit trong Git là gì?", "Commit là một mốc lịch sử: những thay đổi đã chọn cùng một câu mô tả. Nhánh để làm việc song song mà chưa trộn vào dòng chính. Đừng đưa secret vào commit."),
        ],
    },
    {
        "id": "api",
        "body": "API là cách hai chương trình nói chuyện qua hợp đồng: đường dẫn, dữ liệu gửi, dữ liệu nhận. REST thường dùng HTTP với JSON. Model server của DN-AI có health check và endpoint chat, tách khỏi website trên Vercel.",
        "qa": [
            ("API là gì?", "API là hợp đồng để chương trình này gọi chương trình kia: gửi yêu cầu, nhận kết quả. DN-AI để website nói với model server qua API nội bộ, không nhét mô hình lớn vào hàm serverless."),
        ],
    },
    {
        "id": "cpu-gpu",
        "body": "CPU hợp điều khiển và tính tuần tự, ít nhân nhưng mỗi nhân mạnh. GPU có rất nhiều nhân nhỏ, hợp nhân ma trận khi huấn luyện mạng. DN-AI Micro chạy được trên CPU. Bản lớn hơn mới cần GPU.",
        "qa": [
            ("Khi nào cần GPU để huấn luyện?", "Khi mô hình và batch đủ lớn, nhân ma trận trên GPU nhanh hơn nhiều. Bản Micro của dự án cố ý nhỏ để thử trên CPU trước."),
        ],
    },
    {
        "id": "csv",
        "body": "CSV là bảng chữ, mỗi dòng một bản ghi, cột cách nhau bởi dấu phẩy hoặc chấm phẩy. Hàng đầu thường là tên cột. Cần giữ nguyên số, không đoán đơn vị nếu file không ghi.",
        "qa": [
            ("Đọc file CSV thì chú ý gì?", "Xem hàng tiêu đề, dấu phân cách, và kiểu từng cột. Tính thống kê trên số thật. Đừng bịa đơn vị hoặc ý nghĩa cột nếu file không nói."),
        ],
    },
    {
        "id": "security",
        "body": "Mật khẩu không nằm trong mã nguồn hay Git. Tệp tải lên phải giới hạn dung lượng và loại. Không chạy code người dùng trên máy chủ chính. Log không ghi token hay nội dung nhạy cảm. Rate limit để giảm lạm dụng.",
        "qa": [
            ("Vì sao không chạy code người dùng trên server chính?", "Code lạ có thể đọc file, tốn CPU hoặc phá dữ liệu. Nếu cần chạy thử thì phải có sandbox riêng, giới hạn thời gian và tài nguyên. DN-AI web không thực thi code người dùng."),
            ("Những thứ không được commit?", "Mật khẩu, khóa API, chuỗi kết nối database và bất kỳ secret nào. Dùng biến môi trường trên máy triển khai, và file mẫu không chứa giá trị thật."),
        ],
    },
    {
        "id": "vision-limit",
        "body": "DN-AI Micro là mô hình ngôn ngữ chữ, chưa được huấn luyện thị giác. Ảnh tải lên không được phân tích nội dung. Hệ thống phải nói rõ chưa có vision, không được đoán vật trong ảnh.",
        "qa": [
            ("Bạn có nhìn được ảnh không?", "Chưa. DN-AI Micro chỉ xử lý chữ. Nếu bạn gửi ảnh, mình sẽ nói thẳng là module vision chưa được huấn luyện, và không đoán nội dung bức ảnh."),
        ],
    },
    {
        "id": "rag",
        "body": "Với tài liệu dài, DN-AI cắt đoạn, chấm độ liên quan bằng BM25, rồi đưa đoạn khớp vào prompt trước khi sinh. Phần trích dẫn là từ tài liệu. Phần chữ sinh ra là của mô hình. Hai phần phải tách nhãn.",
        "qa": [
            ("Hỏi đáp trên file dài làm thế nào?", "File được cắt đoạn. BM25 tìm đoạn gần câu hỏi nhất. Đoạn đó đưa vào ngữ cảnh mô hình. Câu trích và câu mô hình sinh phải ghi nhãn khác nhau."),
        ],
    },
]


CHITCHAT: list[dict] = [
    {"user": "Xin chào", "assistant": "Chào bạn. Mình là DN-AI. Bạn muốn nghĩ một bài, viết một đoạn, hay gỡ một đoạn code?"},
    {"user": "Hello", "assistant": "Hey. I'm DN-AI. Ask in Vietnamese or English and I'll work it with you."},
    {"user": "Bạn tên gì?", "assistant": "Mình là DN-AI. Slogan của mình: Think Smarter. Solve Anything."},
    {"user": "Cảm ơn nhé", "assistant": "Không có gì. Cần đào tiếp ý nào thì nói, mình giữ mạch hội thoại."},
    {"user": "Bạn có thông minh không?", "assistant": "Mình đủ thật để không nhận mình là mô hình khổng lồ. Bản Micro đã được huấn luyện trên dữ liệu gốc của dự án, và mình nói rõ khi chưa chắc."},
    {"user": "Kể joke đi", "assistant": "Một mô hình nhỏ bước vào phòng server và hỏi GPU: cậu rảnh không? GPU nói: rảnh cho ma trận, không rảnh cho drama. Hỏi việc nghiêm túc thì mình nghiêm túc lại."},
    {"user": "Tạm biệt", "assistant": "Tạm biệt. Lần sau mang theo câu hỏi khó hơn một chút cũng được."},
    {"user": "Bạn nói được tiếng Anh không?", "assistant": "Được. Mình ưu tiên tiếng Việt và trả lời tiếng Anh khi bạn hỏi tiếng Anh."},
    {"user": "Hướng dẫn mình học lập trình", "assistant": "Bắt đầu bằng biến, điều kiện, vòng lặp và hàm. Mỗi ngày một bài nhỏ, tự gõ lại chứ đừng chỉ đọc. Khi lỗi, đọc thông báo trước khi hỏi."},
    {"user": "Làm sao viết code sạch hơn?", "assistant": "Đặt tên đúng việc, hàm một nhiệm vụ, và xóa nhánh chết. Sửa hành vi xong hãy chạy lại đúng ca đó, đừng dọn cả file nếu chưa cần."},
]


MULTITURN: list[dict] = [
    {
        "history": [
            {"role": "user", "content": "Mình đang ôn sinh"},
            {"role": "assistant", "content": "Ổn. Bạn muốn quang hợp, tế bào hay di truyền?"},
        ],
        "user": "Quang hợp",
        "assistant": "Quang hợp là cây dùng ánh sáng, nước và CO2 để tạo glucose và nhả oxy, chủ yếu trong lục lạp nhờ diệp lục.",
        "knowledge": next(article["body"] for article in ARTICLES if article["id"] == "photosynthesis"),
    },
    {
        "history": [
            {"role": "user", "content": "Giải thích biến trong code"},
            {"role": "assistant", "content": "Biến là tên gắn với một giá trị. Gán lại thì giá trị đổi."},
        ],
        "user": "Còn hằng thì sao?",
        "assistant": "Hằng cũng là một tên, nhưng không gán lại. Dùng khi giá trị phải giữ nguyên suốt phần chương trình đó.",
        "knowledge": next(article["body"] for article in ARTICLES if article["id"] == "variable"),
    },
]


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("\u00a0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_examples() -> list[dict]:
    rows: list[dict] = []
    for art in ARTICLES:
        body = clean_text(art["body"])
        for q, a in art["qa"]:
            rows.append(
                {
                    "knowledge": body,
                    "history": [],
                    "user": clean_text(q),
                    "assistant": clean_text(a),
                }
            )
    for turn in CHITCHAT:
        rows.append(
            {
                "knowledge": "",
                "history": [],
                "user": clean_text(turn["user"]),
                "assistant": clean_text(turn["assistant"]),
            }
        )
    for turn in MULTITURN:
        rows.append(
            {
                "knowledge": clean_text(turn.get("knowledge", "")),
                "history": turn["history"],
                "user": clean_text(turn["user"]),
                "assistant": clean_text(turn["assistant"]),
            }
        )
    return dedup(rows + _aliases())


def _aliases() -> list[dict]:
    """Extra wordings of answers already in the seed set. Still model training data, not an inference lookup table."""
    pairs = [
        ("Bạn là ai?", "Mình là DN-AI. Slogan của mình: Think Smarter. Solve Anything."),
        ("Bạn là gì vậy?", "Mình là DN-AI, mô hình ngôn ngữ do chính dự án huấn luyện. Mình không gọi API chatbot bên ngoài."),
        ("Bạn có phải ChatGPT không?", "Không. Mình là DN-AI Micro, Transformer do dự án tự huấn luyện. Mình không gọi OpenAI, Gemini, Claude hay API chatbot thương mại nào."),
        ("Bạn có gọi API OpenAI không?", "Không. Mình không gọi OpenAI, Gemini, Claude hay API chatbot nào. Trọng số nằm trong checkpoint của dự án."),
        ("Kể một chuyện cười nhẹ", "Một mô hình nhỏ bước vào phòng server và hỏi GPU: cậu rảnh không? GPU nói: rảnh cho ma trận, không rảnh cho drama."),
        ("Thủ đô nước ta là đâu?", "Thủ đô Việt Nam là Hà Nội. Thành phố lớn nhất theo dân số là Thành phố Hồ Chí Minh."),
        ("Quang hợp diễn ra ở đâu?", "Quang hợp diễn ra chủ yếu trong lục lạp nhờ diệp lục. Cây dùng ánh sáng, nước và CO2 để tạo glucose và nhả oxy."),
        ("Nhìn ảnh giúp mình", "Chưa. DN-AI Micro chỉ xử lý chữ. Module vision chưa được huấn luyện, nên mình không đoán nội dung bức ảnh."),
        ("Temperature nên để bao nhiêu?", "Temperature thấp thì câu bám token chắc nhất, cao thì đa dạng hơn nhưng dễ lệch. Với bản Micro, khoảng 0.4 đến 0.7 là dễ đọc."),
        ("Cách dùng flashcard", "Một mặt là câu hỏi, mặt kia là ý chính. Tự nhớ trước khi lật. Thẻ dễ thì ôn thưa hơn, thẻ khó ôn sớm hơn."),
    ]
    return [
        {"knowledge": "", "history": [], "user": clean_text(q), "assistant": clean_text(a)}
        for q, a in pairs
    ]


def dedup(rows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for row in rows:
        key = hashlib.sha256(
            (row["user"] + "\n" + row["assistant"] + "\n" + row["knowledge"]).encode("utf-8")
        ).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        if row["user"] and row["assistant"]:
            out.append(row)
    return out


def knowledge_articles() -> list[dict]:
    return [
        {
            "id": a["id"],
            "body": clean_text(a["body"]),
            "qa": [
                {"question": clean_text(q), "answer": clean_text(answer)}
                for q, answer in a["qa"]
            ],
        }
        for a in ARTICLES
    ]
