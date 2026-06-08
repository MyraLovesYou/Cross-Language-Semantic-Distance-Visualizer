from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

english = [
    "The cat is on the table",
    "It's a piece of cake.",
    "Could you please confirm this?",
    "I study once in a blue moon.",
]

japanese = [
    "猫はテーブルの上にいます。",
    "朝飯前です。",
    "ご確認よろしくお願いします？",
    "めったに勉強しない。",
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