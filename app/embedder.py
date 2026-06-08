from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

english = [
    "The cat is on the table",
    "It's a piece of cake.",
    "Could you please confirm this?",
]

japanese = [
    "猫はテーブルの上にいます",
    "朝飯前です",
    "ご確認よろしくお願いします",
]

embeddings1 = model.encode(english)
embeddings2 = model.encode(japanese)

# Compute cosine similarities
similarities = model.similarity(embeddings1, embeddings2)

for idx_i, sentence1 in enumerate(english):
    print(sentence1)
    print(f" - {japanese[idx_i]}: {similarities[idx_i][idx_i]:.4f}")