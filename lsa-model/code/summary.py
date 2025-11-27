import numpy as np
import re
from underthesea import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def summarize_LSA(text, summary_ratio=0.5, max_features=5000, n_components=3):
    # ====== PREPROCESS ======
    text = text.lower()
    
    def protect_numbers(t):
        return re.sub(r'(\d+)\.(\d+)', r'\1_\2', t)

    def restore_numbers(t):
        return t.replace('_', '.')

    text_safe = protect_numbers(text)
    sentences = sent_tokenize(text_safe)
    sentences = [restore_numbers(s) for s in sentences]
    sentences_tokenized = [word_tokenize(s, format="text") for s in sentences]

    # ====== TF-IDF ======
    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(sentences_tokenized)

    # ====== LSA ======
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    lsa_matrix = svd.fit_transform(tfidf_matrix)

    # ====== SCORE ======
    scores = np.sum(np.abs(lsa_matrix), axis=1)

    # số câu cần lấy
    k = max(4, int(len(sentences) * summary_ratio))

    # indices của câu quan trọng nhất
    top_idx = np.argsort(scores)[-k:]
    top_idx = sorted(top_idx)  # giữ đúng thứ tự câu gốc

    # ====== BUILD SUMMARY ======
    summary = " ".join(sentences[i] for i in top_idx)
    return summary

document = "Dự án tái thiết Sao Hỏa (Mars Terraforming Project) đã bước vào giai đoạn thứ 3 sau 15 năm chuẩn bị miệt mài. Mục tiêu chính là nâng nhiệt độ trung bình của hành tinh lên thêm 5 độ C trong vòng hai thập kỷ tiếp theo. Hiện tại, trạm nghiên cứu chính, có tên gọi là Ares Base 4, đang duy trì một đội ngũ gồm 128 nhà khoa học và kỹ sư làm việc liên tục. Mỗi ngày, họ thực hiện ít nhất 300 thí nghiệm khác nhau liên quan đến khí quyển và địa chất. Chỉ riêng tuần trước, 24 chuyến bay không người lái đã được phóng để gieo hạt giống vi khuẩn quang hợp biến đổi gen. 90% năng lượng cần thiết cho căn cứ được cung cấp bởi 8 cánh đồng pin mặt trời khổng lồ, trải rộng trên diện tích gần 10 km vuông. Các nhà khoa học ước tính rằng, cần phải vận chuyển khoảng 500 tấn vật liệu mỗi tháng từ Trái Đất để duy trì tiến độ. Tuy nhiên, tỷ lệ thành công của các nhiệm vụ vận chuyển hiện tại đã đạt mức đáng kinh ngạc là 99.7%. Trong phòng thí nghiệm, họ vừa phát hiện ra rằng 7 loại khoáng chất quý hiếm có thể được sử dụng để ổn định lớp vỏ hành tinh. Chi phí cho toàn bộ dự án đã vượt mốc 20 nghìn tỷ đô la Mỹ, nhưng lợi ích tiềm năng là vô giá. Chỉ còn 4 năm nữa là đến thời điểm phóng mô-đun sinh học đầu tiên mang theo 1000 loài thực vật chịu lạnh đặc biệt. Nhiệm vụ này được dự kiến sẽ diễn ra vào ngày 25 tháng 3 năm 2060. Các máy dò robot đã khoan sâu xuống tới 800 mét dưới bề mặt để tìm kiếm các túi nước ngầm cổ đại. Họ đã tìm thấy bằng chứng về một hồ nước có khả năng chứa tới 1.2 triệu mét khối nước đóng băng. 2 nhà khoa học kỳ cựu tin rằng, với tốc độ này, những cư dân đầu tiên có thể đặt chân lên Sao Hỏa trong vòng 50 năm tới."
summary = summarize_LSA(document)

# ==== GHI RA FILE ====
output_path = "D:\\Introduction to AI\\Project-AI\\lsa-model\\data\\single_summary.txt"

with open(output_path, "w", encoding="utf-8") as f:
    f.write("===== ORIGINAL DOCUMENT =====\n")
    f.write(document + "\n\n")
    
    f.write("===== SUMMARY (LSA) =====\n")
    f.write(summary + "\n")

print("Đã ghi xong vào file:", output_path)