1. Kiến trúc tổng quát

Mô hình abstractive summarization = Encoder – Decoder:

Encoder: đọc văn bản đầu vào (paper text → sequence of tokens).

Decoder: sinh ra summary (chuỗi token ngắn gọn hơn).

Attention: giúp decoder tập trung vào các phần quan trọng trong văn bản.

📌 Một kiến trúc kinh điển:

Encoder = LSTM/GRU

Decoder = LSTM + Attention

(Tuỳ chọn nâng cao: Transformer Encoder–Decoder)

2. Dữ liệu

Bạn cần dataset có (input text, summary). Một số lựa chọn:

CNN/DailyMail dataset (tin tức + highlights).

XSum (BBC news + one-sentence summary).

ArXiv / PubMed dataset (tóm tắt bài báo khoa học).

Nếu không muốn dataset lớn → bạn tự làm dataset nhỏ từ một vài paper (copy phần abstract làm summary).

3. Pipeline huấn luyện
Bước 1. Tiền xử lý

Tokenize input & output.

Tạo vocabulary (word2index, index2word).

Padding + truncation cho chuỗi dài.

Bước 2. Xây model Seq2Seq

Ví dụ với PyTorch:

import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        embed = self.embedding(x)
        outputs, (h, c) = self.lstm(embed)
        return outputs, (h, c)


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size*2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: (batch, hidden_size)
        # encoder_outputs: (batch, seq_len, hidden_size)
        seq_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size + hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*2, vocab_size)
        self.attention = Attention(hidden_size)

    def forward(self, x, hidden, cell, encoder_outputs):
        x = x.unsqueeze(1)
        embed = self.embedding(x)
        attn_weights = self.attention(hidden[-1], encoder_outputs)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        lstm_input = torch.cat((embed, attn_applied), dim=2)
        outputs, (h, c) = self.lstm(lstm_input, (hidden, cell))
        out = self.fc(torch.cat((outputs.squeeze(1), attn_applied.squeeze(1)), dim=1))
        return out, h, c, attn_weights

4. Huấn luyện

Loss: nn.CrossEntropyLoss(ignore_index=PAD_IDX).

Optimizer: Adam.

Teacher forcing trong training (cho decoder thấy ground-truth token).

Huấn luyện vài epoch (dataset nhỏ).

5. Sinh tóm tắt (Inference)

Cho input vào Encoder → lấy hidden states.

Decoder sinh token từng bước (greedy search hoặc beam search).

Dừng khi gặp <EOS>.

6. Đánh giá

Metric phổ biến: ROUGE (ROUGE-1, ROUGE-2, ROUGE-L).

So sánh summary sinh ra với reference.

7. Tối giản cho mini-project

Nếu thời gian ngắn, bạn có thể:

Train trên một subset nhỏ (vd: 10k samples từ CNN/DailyMail).

Giới hạn vocab (~20k words).

Trình bày rõ kiến trúc Seq2Seq + Attention.

Demo inference trên 1–2 paper thực tế.