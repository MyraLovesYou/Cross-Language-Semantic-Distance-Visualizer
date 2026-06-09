import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


class VectorEngine:
    def __init__(self, model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        load_dotenv()
        hf_token = os.getenv("HF_TOKEN")
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', token=hf_token)
    def generate_embeddings(self, sentences):
        embeddings = self.model.encode(sentences)
        return embeddings


