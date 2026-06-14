import pandas as pd
import numpy as np
import sqlite3
from src.embedder import VectorEngine
from src.processor import generate_similarity_matrix, compress_umap_coordinates

def init_database():
    print("Initializing database...")
    df = pd.read_excel("data/raw/english-japanese-raw-parallel.xlsx")
    engine = VectorEngine()
    en_embeddings = engine.generate_embeddings(df['english'].tolist())
    ja_embeddings = engine.generate_embeddings(df['japanese'].tolist())
    df['similarity_scores'] = generate_similarity_matrix(en_embeddings, ja_embeddings)
    all_vectors = np.vstack([en_embeddings, ja_embeddings])
    coords = compress_umap_coordinates(all_vectors)
    df['en_x'] = coords[:len(df), 0]
    df['en_y'] = coords[:len(df), 1]
    df['ja_x'] = coords[len(df):, 0]
    df['ja_y'] = coords[len(df):, 1]
    conn = sqlite3.connect("data/processed/semantic_space.db")
    df.to_sql("translations", conn, if_exists="replace", index=True, index_label="id")
    conn.close()
    print("Database built.")