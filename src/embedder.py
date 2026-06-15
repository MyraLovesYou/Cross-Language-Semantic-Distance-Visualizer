import os
from sentence_transformers import SentenceTransformer
import streamlit as st

class VectorEngine:
    def __init__(self, model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        hf_token = st.secrets["HF_TOKEN"]
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', token=hf_token)
    def generate_embeddings(self, sentences):
        embeddings = self.model.encode(sentences)
        return embeddings


