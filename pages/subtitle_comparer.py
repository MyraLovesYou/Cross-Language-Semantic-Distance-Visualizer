import streamlit as st
import pandas as pd
import numpy as np
import pysrt
import io
from src.embedder import VectorEngine
from src.processor import generate_similarity_matrix, compress_umap_coordinates
from src.text_cleaner import clean_subtitle_text

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

    aligned_data = []

    for en_sub in en_subs:
    
        en_start = en_sub.start.ordinal # Timestamp converted to milliseconds
        
        best_match = min(ja_subs, key=lambda x: abs(x.start.ordinal - en_start))
        
        if abs(best_match.start.ordinal - en_start) < 2000:
            aligned_data.append({
                "timestamp": str(en_sub.start),
                "english": en_sub.text,
                "japanese": best_match.text
            })
    
    df_subs = pd.DataFrame(aligned_data)
    if not df_subs.empty:
        st.success(f"Successfully aligned {len(df_subs)} dialogue blocks by timestamp proximity!")
        with st.spinner("Analyzing similarity of files..."):
            engine = VectorEngine()
            df_subs['clean_en'] = df_subs["english"].apply(clean_subtitle_text)
            df_subs['clean_ja'] = df_subs["japanese"].apply(clean_subtitle_text)
            en_embeddings = engine.generate_embeddings(df_subs['clean_en'].tolist())
            ja_embeddings = engine.generate_embeddings(df_subs['clean_ja'].tolist())
            df_subs['similarity_score'] = generate_similarity_matrix(en_embeddings, ja_embeddings)

        avg_similarity = df_subs['similarity_score'].mean()
        col3, col4 = st.columns([2,3], vertical_alignment="center")
        with col3:
            st.metric(label="Overall Subtitle Translation Alignment Metric", value=f"{100 * avg_similarity:.1f}% Match")
        with col4:
            st.progress(float(avg_similarity))
  
        columns_to_show = ["clean_ja", "clean_en", "similarity_score"]
        st.dataframe(df_subs[columns_to_show].sort_values(by="similarity_score", ascending=False))

    else:
        st.error("Could not automatically match subtitle together. Make sure to use two parallel srts from the same media.")