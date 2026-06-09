import pandas as pd
import numpy as np
import sqlite3
from src.embedder import VectorEngine
from src.processor import generate_similarity_matrix, compress_umap_coordinates

def init_database():
    print("Initializing database...")
    df = pd.read_excel("data/raw/english-japanese-raw-parallel")
    engine = VectorEngine()
    en_embeddings = engine.generate_embeddings(df['english'].tolist())
    ja_embeddings = engine.generate_embeddings(df['japanese'].tolist())
    df['similarity_scores'] = generate_similarity_matrix(en_embeddings, ja_embeddings)
    all_vectors = np.vstack([en_embeddings, ja_embeddings])
    all_coords = compress_umap_coordinates(all_vectors)