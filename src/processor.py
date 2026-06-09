from sklearn.metrics.pairwise import cosine_similarity
import umap
import numpy as np

def generate_similarity_matrix(embeddings_1, embeddings_2):
    similarity_matrix = cosine_similarity(embeddings_1, embeddings_2)
    diagonal_points = np.diag(similarity_matrix)
    return diagonal_points

def compress_umap_coordinates(all_vectors):
    reducer = umap.UMAP(n_components=2, random_state=42)
    coordinate_matrix = reducer.fit_transform(all_vectors)
    return coordinate_matrix
