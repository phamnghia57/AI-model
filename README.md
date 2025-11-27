# Vietnamese Text Summarization Project — BARTPho Fine-tuned

## Mô tả dự án
Dự án này sử dụng mô hình **BARTPho** fine-tuned để thực hiện tóm tắt văn bản tiếng Việt.  
Hệ thống được chia thành 4 module chính:

1. **Prepare_data**: Bước đầu tiền xử lý file data.csv
2. **Training**: Huấn luyện mô hình BARTPho trên dữ liệu tóm tắt.
3. **Inference**: Sinh tóm tắt cho văn bản mới.
4. **Evaluate**: Đánh giá chất lượng mô hình với các metric tiêu chuẩn (ROUGE, BERTScore, Cosine similarity).
5. **App**: Giao diện trực quan Streamlit để upload file hoặc nhập URL và nhận tóm tắt.

## Cách sử dụng

Open cmd -> cd '/thu muc project/'
python -m venv bartpho_env
bartpho_env\Scripts\activate
pip install -r requirements.txt
python scripts/prepapre_data.py
python scripts/train.py vinai/bartpho-word data/processed outputs/bartpho-finetuned
python scripts/evaluate_model.py
python scripts/infer.py outputs/bartpho-finetuned "TEXT"
streamlit run app.py