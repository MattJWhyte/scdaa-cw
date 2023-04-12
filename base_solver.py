
import numpy as np
from lqr_solver import LQR_Solver

H = np.eye(2)
M = np.eye(2)
sigma = 0.05 * np.eye(2)
T = 1
C = 0.1 * np.eye(2)
D = 0.1 * np.eye(2)
R = np.eye(2)

solver = LQR_Solver(H, M, sigma, T, C, D, R)

