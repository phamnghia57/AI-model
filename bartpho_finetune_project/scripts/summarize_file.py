"""
File Summary Generator using Fine-tuned BARTpho

1. Tải mô hình BARTpho đã fine-tuned.
2. Đọc văn bản từ file PDF hoặc DOCX.
3. Sinh tóm tắt tự động bằng mô hình Seq2Seq.
"""

import sys
import os
import pdfplumber
import docx
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model và tokenizer

# Đường dẫn chứa checkpoint mô hình đã fine-tuned
model_dir = "outputs/bartpho-finetuned"  # thay bằng checkpoint của bạn

# Tải tokenizer và mô hình từ thư mục fine-tuned
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


# Hàm đọc docx
def read_docx(file_path):
    """
    Đọc nội dung từ file DOCX và ghép toàn bộ paragraph thành một chuỗi.

    Args:
        file_path (str): đường dẫn file .docx

    Returns:
        str: toàn bộ nội dung văn bản dạng string
    """
    doc = docx.Document(file_path)
    full_text = []
    # Lặp qua từng đoạn văn, loại bỏ đoạn rỗng
    for para in doc.paragraphs:
        if para.text.strip():
            full_text.append(para.text)
    return "\n".join(full_text)


# Hàm đọc pdf
def read_pdf(file_path):
    """
    Đọc nội dung PDF page-by-page bằng pdfplumber.

    Args:
        file_path (str): đường dẫn file .pdf

    Returns:
        str: text trích xuất từ toàn bộ các trang PDF
    """
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text


# Hàm tóm tắt
def summarize(text):
    """
    Sinh tóm tắt từ một đoạn văn bản đầu vào.

    Args:
        text (str): nội dung cần tóm tắt

    Returns:
        str: kết quả tóm tắt tiếng Việt
    """
    # Tokenize đầu vào — đảm bảo cắt chuỗi nếu quá dài
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(device)

    ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=256,
        min_length=30,
        num_beams=5
    )
    return tokenizer.decode(ids[0], skip_special_tokens=True)


def main():
    """
    Hàm chính:
    - Kiểm tra tham số dòng lệnh
    - Nhận đường dẫn file
    - Đọc file PDF/DOCX
    - Sinh summary
    - In kết quả
    """
    if len(sys.argv) < 2:
        print("Usage: python summarize_file.py <path_to_file>")
        return

    file_path = sys.argv[1]

    if not os.path.exists(file_path):
        print("File không tồn tại.")
        return

    # đọc file tùy vào định dạng
    if file_path.endswith(".docx"):
        text = read_docx(file_path)
    elif file_path.endswith(".pdf"):
        text = read_pdf(file_path)
    else:
        print("Chỉ hỗ trợ .docx hoặc .pdf")
        return

    # Sinh tóm tắt
    summary = summarize(text)

    print("\n===== TÓM TẮT =====")
    print(summary)


if __name__ == "__main__":
    main()
