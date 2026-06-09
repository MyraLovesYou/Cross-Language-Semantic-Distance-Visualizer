import pandas as pd
import numpy as np
import sqlite3
from src.embedder import VectorEngine
from src.processor import generate_similarity_matrix, compress_umap_coordinates

def init_database():
    print("Initializing database...")
    df = pd.read_excel("data/raw/english-japanese-raw-parallel")
    