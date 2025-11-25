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

document = "Người đầu tiên xứng với danh xưng hảo hán không ai khác chính la Võ Tòng. Cả đời Võ Tòng hành hiệp trượng nghĩa, thích bênh vực kẻ yếu, hoàn toàn giống với một người anh hùng. Hơn nữa Võ Tòng còn là một người yêu hận phân minh, không tàn sát người vô tội, đây là điểm rất quan trọng. Lấy chuyện Võ Tòng báo thù cho Võ Đại Lang làm ví dụ, Võ Tòng chỉ giết đúng hai kẻ liên quan trực tiếp là Phan Kim Liên và Tây Môn Khánh, Võ Tòng không hề ra tay với Vương bà dù bà ta cũng là người góp phần không nhỏ, đối với những người đi hóng chuyện, Võ Tòng cũng không ra tay thảo phạt hàng loạt, làm người rất có chừng mực, cho thấy bản chất đúng đắn của một anh hùng hảo hán. Đứng ở góc độ hiện tại, nhiều người có thể cảm thấy cách cư xử của Võ Tòng là điều hết sức bình thường, bởi lẽ chuyện này quả thực chẳng liên quan tới những người khác, nhưng đây lại chỉ là tư duy của người hiện đại chúng ta, trong Thủy Hử, không thiếu gì hảo hán chẳng cần biết vô tội hay không, chỉ cần động thủ là phải đánh cho tới bến, cho sảng khoái thì thôi. Nếu đổi lại là Lý Quỳ, có lẽ người bị giết sẽ nhiều hơn, chỉ cần là những người biết chuyện của Võ Đại Lang mà không chịu lên tiếng hay làm gì, e là có lẽ đều sẽ bị giết… Nhưng Võ Tòng thì khác, sau khi biết đầu đuôi câu chuyện, Võ Tòng chỉ oan có đầu nợ có chủ, chỉ nhắm vào đương sự, không lạm dụng vũ lực, đây mới chính là anh hùng hảo hán. Thủy Hử 108 anh hùng, chỉ 4 người thực sự là hảo hán gồm những ai."
summary = summarize_LSA(document)

# ==== GHI RA FILE ====
output_path = "D:\\Introduction to AI\\Project-AI\\active\\data\\single_summary.txt"

with open(output_path, "w", encoding="utf-8") as f:
    f.write("===== ORIGINAL DOCUMENT =====\n")
    f.write(document + "\n\n")
    
    f.write("===== SUMMARY (LSA) =====\n")
    f.write(summary + "\n")

print("Đã ghi xong vào file:", output_path)