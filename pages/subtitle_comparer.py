import streamlit as st
import pandas as pd
import numpy as np
import pysrt
import io
from src.embedder import VectorEngine
from src.processor import generate_similarity_matrix, compress_umap_coordinatess

st.title("Subtitle Translation Analyzer")
st.write("Please upload an English and Japanese .srt file to compare subtitles")

col1, col2 = st.columns(2)
with col1:
    en_file = st.file_uploader("Upload English SRT", type=["srt"])
with col2:
    ja_file = st.file_uploader("Upload Japanese SRT", type=["srt"])

if en_file and ja_file:
    en_stream = io.StringIO(en_file.getvalue().decode("utf-8"))
    ja_stream = io.StringIO(ja_file.getvalue().decode("utf-8"))

    en_subs = pysrt.from_string(en_stream.read())
    ja_subs = pysrt.from_string(ja_stream.read())