import streamlit as st
import pandas as pd
import numpy as np
import pysrt
import io
from src.embedder import VectorEngine
from src.processor import generate_similarity_matrix, compress_umap_coordinatess