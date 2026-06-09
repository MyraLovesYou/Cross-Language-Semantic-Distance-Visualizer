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
    "I study once in a blue moon.",
    "The frog in the well doesn't know about the vast ocean",
]

japanese = [
    "猫はテーブルの上にいます。",
    "朝飯前です。",
    "ご確認よろしくお願いします？",
    "めったに勉強しない。",
    "虎穴に入らずんば虎子を得ず",
]

sentences = [
    "The cat is on the table",
    "It's a piece of cake.",
    "Could you please confirm this?",
    "猫はテーブルの上にいます",
    "朝飯前です",
    "ご確認よろしくお願いします",
]
embeddings1 = model.encode(english)
embeddings2 = model.encode(japanese)

# Compute cosine similarities
similarities = cosine_similarity(embeddings1, embeddings2)

for idx_i, sentence1 in enumerate(english):
    print(sentence1)
    print(f" - {japanese[idx_i]}: {similarities[idx_i][idx_i]:.4f}")