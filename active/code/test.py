import pandas as pd
import numpy as np
import re
from underthesea import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from rouge import Rouge
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("D:\\Introduction to AI\\Project-AI\\bartpho_finetune_project\\data\\processed\\test.csv")

df['text'] = df['text'].str.lower()

def protect_numbers(text):
    return re.sub(r'(\d+)\.(\d+)', r'\1_\2', text)

def restore_numbers(text):
    return text.replace('_', '.')

df['texts_safe'] = df['text'].apply(protect_numbers)

df['texts_sentences'] = df['texts_safe'].apply(lambda x: sent_tokenize(x))

df['texts_sentences'] = df['texts_sentences'].apply(
    lambda sents: [restore_numbers(sent) for sent in sents]
)

df['texts_sentences_tokenized'] = df['texts_sentences'].apply(
    lambda sents: [word_tokenize(sent, format="text") for sent in sents]
)

all_sentences = [sentence for doc in df['texts_sentences_tokenized'] for sentence in doc]

vectorizer = TfidfVectorizer(
    max_features=5000, 
    stop_words=None    
)

tfidf_matrix = vectorizer.fit_transform(all_sentences)

k = 5

svd = TruncatedSVD(n_components=k, random_state=42)

lsa_matrix = svd.fit_transform(tfidf_matrix)

sentence_scores = np.sum(np.abs(lsa_matrix), axis=1)

summary_ratio = 0.3 

summaries = []

for doc_idx, sentences in enumerate(df['texts_sentences_tokenized']):
    start_idx = sum(df['texts_sentences_tokenized'].iloc[:doc_idx].apply(len))
    end_idx = start_idx + len(sentences)
    scores = sentence_scores[start_idx:end_idx]

    k = max(1, int(len(sentences) * summary_ratio))

    top_sentence_indices = np.argsort(scores)[-k:]

    top_sentence_indices = sorted(top_sentence_indices)

    summary = ' '.join([sentences[i] for i in top_sentence_indices])
    summaries.append(summary)

df['Summary_LSA'] = summaries

for i in range(4):
    print("Original Document:\n", df['text'].iloc[i][:300], "...\n")
    print("Summary:\n", df['Summary_LSA'].iloc[i], "\n")
    print("="*80)

with open("D:\\Introduction to AI\\Project-AI\\active\\data\\test_data_summaries.txt", "w", encoding="utf-8") as f:
    for i in range(min(1000, len(df))):
        f.write(f"Document {i+1}:\n")
        f.write("Original: " + df['text'].iloc[i] + "\n")
        f.write("Summary: " + df['Summary_LSA'].iloc[i] + "\n")
        f.write("="*80 + "\n")

rouge = Rouge()
scores_list = []

for i in range(len(df)):
    reference = df['summary'].iloc[i]        
    generated = df['Summary_LSA'].iloc[i]     
    scores = rouge.get_scores(generated, reference)[0]
    scores_list.append(scores)

df_rouge = pd.DataFrame(scores_list)
print(df_rouge.head())

rouge_1 = df_rouge['rouge-1'].apply(pd.Series)
rouge_2 = df_rouge['rouge-2'].apply(pd.Series)
rouge_l = df_rouge['rouge-l'].apply(pd.Series)

avg_rouge_1 = rouge_1.mean()
avg_rouge_2 = rouge_2.mean()
avg_rouge_l = rouge_l.mean()

print("Average ROUGE-1:\n", avg_rouge_1)
print("Average ROUGE-2:\n", avg_rouge_2)
print("Average ROUGE-L:\n", avg_rouge_l)

with open("D:\\Introduction to AI\\Project-AI\\active\\data\\test_rouge_scores.txt", "w", encoding="utf-8") as f:
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

cosine_scores = []

for i in range(len(df)):
    texts = [df['Summary_LSA'].iloc[i], df['summary'].iloc[i]]
    vect = TfidfVectorizer().fit_transform(texts)
    score = cosine_similarity(vect[0:1], vect[1:2])[0][0]
    cosine_scores.append(score)

df['cosine_score'] = cosine_scores
print(df['cosine_score'])

print("Average Cosine similarity:", np.mean(cosine_scores))

with open("D:\\Introduction to AI\\Project-AI\\active\\data\\test_cosine_scores.txt", "w", encoding="utf-8") as f:
    f.write("===== Cosine Similarity Scores (per document) =====\n\n")
    for i in range(len(df)):
        f.write(f"Document {i+1}:\n")
        f.write("Cosine similarity: " + str(round(df['cosine_score'].iloc[i], 3)) + "\n")
        f.write("="*24 + "\n")
    
    f.write("\nAverage Cosine similarity: " + str(round(df['cosine_score'].mean(), 3)) + "\n")