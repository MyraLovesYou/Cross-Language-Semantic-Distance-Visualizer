import streamlit as st
import sqlite3
import pandas as pd
import plotly.graph_objects as go
import os
from src.build_database import init_database

DB_PATH = "data/processed/semantic_space.db"
st.set_page_config(layout="wide")

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



st.sidebar.header("Filter Options")
categories = ["All"] + list(df['type'].unique())
selected_category = st.sidebar.selectbox("Linguistic Category", categories)


if selected_category != "All":
    filtered_df = df[df['type'] == selected_category]
else:
    filtered_df = df

fig = go.Figure()

for idx, row in filtered_df.iterrows():
    fig.add_trace(go.Scatter(
        x=[row['en_x'], row['ja_x']],
        y=[row['en_y'], row['ja_y']],
        mode='lines',
        line=dict(color='rgba(100, 116, 139, 0.4)', width=1.5), # Soft gray line
        showlegend=False,
        hoverinfo='none'
    ))

fig.add_trace(go.Scatter(
    x=filtered_df['en_x'],
    y=filtered_df['en_y'],
    mode='markers',
    name='English Anchor',
    marker=dict(color='#001F3F', size=10), 
    text=filtered_df['english'],
    hovertemplate="<b>English:</b> %{text}<extra></extra>"
))

fig.add_trace(go.Scatter(
    x=filtered_df['ja_x'],
    y=filtered_df['ja_y'],
    mode='markers',
    name='Japanese Translation',
    marker=dict(color='#D4AF37', size=10),
    text=filtered_df['japanese'],
    hovertemplate="<b>Japanese:</b> %{text}<extra></extra>"
))

fig.update_layout(
    title="2D UMAP Map of Aligned Vector Embeddings",
    xaxis_title="Abstract UMAP Dimension X",
    yaxis_title="Abstract UMAP Dimension Y",
    hovermode='closest',
    width=900,
    height=600,
    template="plotly_white"
)

col1, col2 = st.columns([3, 1])

with col1:
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.write("### Data Breakdown")
    st.write(f"Showing **{len(filtered_df)}** parallel phrase pairs.")
    st.write("#### Highest Similarity in View")
    high_drift = filtered_df.sort_values(by='similarity_scores', ascending=False)[['english', 'similarity_scores']].head(5)
    st.dataframe(high_drift, hide_index=True)