import numpy as np

def compute_cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two feature vectors.
    
    Args:
        vec1 (np.ndarray): First feature vector.
        vec2 (np.ndarray): Second feature vector.
    
    Returns:
        float: Cosine similarity value.
    """
    vec1 = np.squeeze(vec1)
    vec2 = np.squeeze(vec2)
    vec1_norm = np.linalg.norm(vec1)
    vec2_norm = np.linalg.norm(vec2)
    if vec1_norm == 0 or vec2_norm == 0:
        return 0.0
    return np.dot(vec1, vec2) / (vec1_norm * vec2_norm)