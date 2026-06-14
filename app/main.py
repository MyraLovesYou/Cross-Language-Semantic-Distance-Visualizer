import streamlit as st
import sqlite3
import pandas as pd
import plotly.graph_objects as go
import os
from src.build_database import init_database

DB_PATH = "data/processed/semantic_space.db"

if not os.path.exists(DB_PATH):
    with st.status("First-time setup detected: Generating vector embeddings and building database...", expanded=True) as status:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        init_database() 
        status.update(label="Database successfully built!", state="complete", expanded=False)


@st.cache_data 
def load_data_from_db():
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT english, japanese, type, similarity_scores, en_x, en_y, ja_x, ja_y FROM translations"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

df = load_data_from_db()