import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', token=hf_token)

english = [
    "The cat is on the table",
    "It's a piece of cake.",
    "Could you please confirm this?",
    "Kill two birds with one stone.",
    "Doing the homework was a piece of cake",
]

japanese = [
    "猫はテーブルの上にいます。",
    "朝飯前です。",
    "ご確認よろしくお願いします？",
    "一石二鳥 ",
    "宿題をするのは朝飯前だった",
]

embeddings1 = model.encode(english)
embeddings2 = model.encode(japanese)

# Compute cosine similarities
similarities = cosine_similarity(embeddings1, embeddings2)

for idx_i, sentence1 in enumerate(english):
    print(sentence1)
    print(f" - {japanese[idx_i]}: {similarities[idx_i][idx_i]:.4f}")