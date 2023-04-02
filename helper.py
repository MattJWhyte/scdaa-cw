
import numpy as np

def positive_def_matrix():
    A = np.random.randn(2,2)
    B = A + A.T
    M = (B.T @ B) / 4.0 + 0.00001 * np.eye(2)
    print("Positive definite matrix ", M)
    return M

