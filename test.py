

from scipy.integrate import odeint
import numpy as np
from lqr_solver import LQR_Solver
import matplotlib.pyplot as plt
import torch
from helper import positive_def_matrix

'''
T = 5
t = np.linspace(T,0,100)

y0 = np.eye(2).flatten()

def der(y,t):
    return np.eye(2).flatten()

S = np.reshape(odeint(der, y0, t)[::-1],(100,2,2))

print(S[0])

print(S[-1])'''

# SET UP LQR ----------------------------------------------

H = np.eye(2) #positive_def_matrix()
M = np.eye(2) #positive_def_matrix() #np.array([[0.15663973 0.15513884],[0.15513884 0.20362521]])

sigma = np.eye(2) #np.array([[0.05, 0.0],[0.05,0.0]])#0.05*np.eye(2) #positive_def_matrix()
T = 1
C = np.eye(2) #positive_def_matrix()
D = np.eye(2) #positive_def_matrix()
R = np.eye(2) #positive_def_matrix()


LQR_Ricatti = LQR_Solver(H, M, sigma, T, C, D, R)


N = 10000
tt = np.linspace(0,T,N+1)

St = LQR_Ricatti.solve_ricatti(tt).numpy()

num_realisations = 10000
x_init = torch.ones((2,1))*3

Out_tt, Out_x = LQR_Ricatti.simulate_X(N, num_realisations, 0, x_init)

dt = T/N

J_est = LQR_Ricatti.evaluate_J_X(Out_tt, Out_x, dt)

print("Last state", Out_x[0,-1])

print("J_est size", J_est.size())
print(Out_x.size())

J_est_mean = torch.mean(J_est)
V_init = LQR_Ricatti.v(Out_tt, Out_x[0].transpose(-1, -2))[0,0]
print("V init", V_init)
print("J est", J_est_mean)

D = J_est_mean-V_init
print(D)
