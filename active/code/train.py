import pandas as pd
import numpy as np
import re
from underthesea import word_tokenize, sent_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from rouge import Rouge
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("D:\\Introduction to AI\\Project-AI\\active\\data\\data-1.csv")

df= df.drop(["Unnamed: 0", "Dataset"], axis=1)

# Chuyển về chữ thường
df['Document'] = df['Document'].str.lower()

def protect_numbers(text):
    return re.sub(r'(\d+)\.(\d+)', r'\1_\2', text)

def restore_numbers(text):
    return text.replace('_', '.')

df['Document_safe'] = df['Document'].apply(protect_numbers)

# Tách câu trước
df['Documents_sentences'] = df['Document_safe'].apply(lambda x: sent_tokenize(x))

# Restore số liệu
df['Documents_sentences'] = df['Documents_sentences'].apply(
    lambda sents: [restore_numbers(sent) for sent in sents]
)

# Tokenize từng câu
df['Documents_sentences_tokenized'] = df['Documents_sentences'].apply(
    lambda sents: [word_tokenize(sent, format="text") for sent in sents]
)

print(df.head())

# Số lượng từ sau khi tokenize
df['length_before'] = df['Document'].apply(lambda x: len(x.split()))
df['length_after'] = df['Documents_sentences_tokenized'].apply(
    lambda sents: sum(len(sent.split()) for sent in sents)
)

plt.figure(figsize=(12,5))
plt.hist(df['length_before'], bins=20, alpha=0.5, label='Before Tokenization')
plt.hist(df['length_after'], bins=20, alpha=0.5, label='After Tokenization')
plt.xlabel('Number of words')
plt.ylabel('Number of documents')
plt.title('Word Count Before and After Tokenization')
plt.legend()
plt.show()

# Top 10 từ xuất hiện nhiều nhất
# Flatten tất cả từ trong các câu đã tokenize
all_words = [word for doc in df['Documents_sentences_tokenized'] for sent in doc for word in sent.split()]

# Loại bỏ punctuation nếu cần
punctuations = {'.', '!', '?', ',', ':', ';', '(', ')', '-', '"', "'"}
all_words = [w for w in all_words if w not in punctuations]

most_common = Counter(all_words).most_common(10)

words, counts = zip(*most_common)

plt.figure(figsize=(8,5))
plt.bar(words, counts)
plt.title('Top 10 frequent words after preprocessing')
plt.ylabel('Frequency')
plt.show()

# Số câu của mỗi văn bản sau khi split
df['num_sentences'] = df['Documents_sentences_tokenized'].apply(len)

plt.figure(figsize=(10,5))
plt.hist(df['num_sentences'], bins=20, alpha=0.7, color='green')
plt.xlabel('Number of sentences per document')
plt.ylabel('Number of documents')
plt.title('Sentence Count per Document after Sentence Splitting')
plt.show()

'''
Biểu diễn câu bằng TF-IDF
'''

# Gộp tất cả câu thành một danh sách
all_sentences = [sentence for doc in df['Documents_sentences_tokenized'] for sentence in doc]

# Khởi tạo TF-IDF
vectorizer = TfidfVectorizer(
    max_features=5000,   # số từ tối đa để giảm ma trận quá lớn
    stop_words=None      # đã xóa stopwords trước đó
)

# Biến tất cả câu thành ma trận TF-IDF
tfidf_matrix = vectorizer.fit_transform(all_sentences)
print(tfidf_matrix.shape)

tfidf_sample = tfidf_matrix[:50,:50].toarray()

plt.figure(figsize=(12,8))
sns.heatmap(tfidf_sample, cmap="YlGnBu")
plt.title("Heatmap TF-IDF (50 câu x 50 từ đầu)")
plt.xlabel("Từ")
plt.ylabel("Câu")
plt.show()

'''
Áp dụng LSA
'''

k = 5

svd = TruncatedSVD(n_components=k, random_state=42)

lsa_matrix = svd.fit_transform(tfidf_matrix)
print(lsa_matrix.shape)

terms = vectorizer.get_feature_names_out()
n_top_words = 10  # số từ quan trọng muốn hiển thị

for topic_idx, topic in enumerate(svd.components_):
    top_terms = sorted(zip(terms, topic), key=lambda x: abs(x[1]), reverse=True)[:n_top_words]
    top_terms = [(word, round(float(weight), 3)) for word, weight in top_terms]
    print(f"Top words for topic {topic_idx}: {top_terms}")
    print()

# Tính score cho từng câu
sentence_scores = np.sum(np.abs(lsa_matrix), axis=1)

# Lấy 10 câu quan trọng nhất
top_indices = np.argsort(sentence_scores)[-10:][::-1]  # sắp xếp giảm dần
for i in top_indices:
    print(all_sentences[i], sentence_scores[i])
    print()

summary_ratio = 0.3  # 30% số câu

# Chứa summary cho từng văn bản
summaries = []

for doc_idx, sentences in enumerate(df['Documents_sentences']):
    # Lấy score của các câu thuộc văn bản này
    start_idx = sum(df['Documents_sentences'].iloc[:doc_idx].apply(len))
    end_idx = start_idx + len(sentences)
    scores = sentence_scores[start_idx:end_idx]

    # Số câu muốn chọn
    k = max(1, int(len(sentences) * summary_ratio))

    # Lấy indices các câu quan trọng nhất
    top_sentence_indices = np.argsort(scores)[-k:]

    # Sắp xếp theo thứ tự xuất hiện trong văn bản
    top_sentence_indices = sorted(top_sentence_indices)

    # Tạo summary
    summary = ' '.join([sentences[i] for i in top_sentence_indices])
    summaries.append(summary)

# Thêm vào dataframe
df['Summary_LSA'] = summaries

# Kiểm tra kết quả
for i in range(4):
    print("Original Document:\n", df['Document'].iloc[i][:300], "...\n")
    print("Summary:\n", df['Summary_LSA'].iloc[i], "\n")
    print("="*80)

rouge = Rouge()

# In văn bản đầu tiên
with open("D:\\Introduction to AI\\Project-AI\\active\\data\\train_data_summaries.txt", "w", encoding="utf-8") as f:
    for i in range(min(1000, len(df))):
        f.write(f"Document {i+1}:\n")
        f.write("Original: " + df['Document'].iloc[i] + "\n")
        f.write("Summary: " + df['Summary_LSA'].iloc[i] + "\n")
        f.write("="*80 + "\n")

rouge = Rouge()
scores_list = []

for i in range(len(df)):
    reference = df['Summary'].iloc[i]          # summary chuẩn
    generated = df['Summary_LSA'].iloc[i]     # summary LSA
    scores = rouge.get_scores(generated, reference)[0]
    scores_list.append(scores)

# Tạo dataframe để xem
df_rouge = pd.DataFrame(scores_list)
print(df_rouge.head())

# Tách các thành phần rouge-1, rouge-2, rouge-l
rouge_1 = df_rouge['rouge-1'].apply(pd.Series)
rouge_2 = df_rouge['rouge-2'].apply(pd.Series)
rouge_l = df_rouge['rouge-l'].apply(pd.Series)

# Tính trung bình
avg_rouge_1 = rouge_1.mean()
avg_rouge_2 = rouge_2.mean()
avg_rouge_l = rouge_l.mean()

# In ra console
print("Average ROUGE-1:\n", avg_rouge_1)
print("Average ROUGE-2:\n", avg_rouge_2)
print("Average ROUGE-L:\n", avg_rouge_l)

# Ghi kết quả ra file
with open("D:\\Introduction to AI\\Project-AI\\active\\data\\train_rouge_scores.txt", "w", encoding="utf-8") as f:
    f.write("===== ROUGE Scores (documents) =====\n")
    f.write(df_rouge.map(
        lambda x: {k: round(v, 3) for k, v in x.items()}
    ).to_string())

    f.write("\n\n===== Average ROUGE-1 =====\n")
    f.write(avg_rouge_1.round(3).to_string())

    f.write("\n\n===== Average ROUGE-2 =====\n")
    f.write(avg_rouge_2.round(3).to_string())

    f.write("\n\n===== Average ROUGE-L =====\n")
    f.write(avg_rouge_l.round(3).to_string())

# cosine
cosine_scores = []

for i in range(len(df)):
    texts = [df['Summary_LSA'].iloc[i], df['Summary'].iloc[i]]  # LSA summary và summary gốc
    vect = TfidfVectorizer().fit_transform(texts)
    score = cosine_similarity(vect[0:1], vect[1:2])[0][0]  # cosine similarity giữa 2 vector
    cosine_scores.append(score)

df['cosine_score'] = cosine_scores
print(df['cosine_score'].head())

print("Average Cosine similarity:", np.mean(cosine_scores))

# Ghi kết quả cosine similarity ra file
with open("D:\\Introduction to AI\\Project-AI\\active\\data\\train_cosine_scores.txt", "w", encoding="utf-8") as f:
    f.write("===== Cosine Similarity Scores (per document) =====\n\n")
    for i in range(len(df)):
        f.write(f"Document {i+1}:\n")
        f.write("Cosine similarity: " + str(round(df['cosine_score'].iloc[i], 3)) + "\n")
        f.write("="*50 + "\n")
    
    f.write("\nAverage Cosine similarity: " + str(round(df['cosine_score'].mean(), 3)) + "\n")

