import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class VectorEngine:
    def __init__(self, model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        load_dotenv()
        hf_token = os.getenv("HF_TOKEN")
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', token=hf_token)
    def generate_embeddings(self, sentences):
        embeddings = self.model.encode(sentences)
        return embeddings


# Compute cosine similarities
# similarities = cosine_similarity(embeddings1, embeddings2)

