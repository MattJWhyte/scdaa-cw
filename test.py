

from scipy.integrate import odeint
import numpy as np
from lqr_solver import LQR_Solver
import matplotlib.pyplot as plt
import torch

'''
T = 5
t = np.linspace(T,0,100)

y0 = np.eye(2).flatten()

def der(y,t):
    return np.eye(2).flatten()

S = np.reshape(odeint(der, y0, t)[::-1],(100,2,2))

print(S[0])

print(S[-1])'''

H = np.zeros((2,2)) #np.eye(2)
M = np.eye(2)
sigma = np.eye(2)
x_init = np.zeros(2)
T = 1
C = np.zeros((2,2))
D = np.eye(2)
R = np.eye(2)

LQR_Ricatti = LQR_Solver(H, M, sigma, x_init, T, C, D, R)

N_T = 100
tt = np.linspace(0,T,N_T)

S = LQR_Ricatti.solve_ricatti(tt)

I = LQR_Ricatti.evaluate_integral(S,tt)

x1 = np.linspace(0,5,N_T)
x2 = np.linspace(3,2,N_T)
xx = torch.zeros((N_T, 1, 2))
xx[:,:,0] = torch.from_numpy(x1).unsqueeze(1)
xx[:,:,1] = torch.from_numpy(x2).unsqueeze(1)


V = LQR_Ricatti.v(torch.from_numpy(tt), xx)

A = LQR_Ricatti.a(torch.from_numpy(tt), xx)

print(V.size())
print(A.size())

N = 5

Out_tt, Out_x = LQR_Ricatti.simulate_X(1000, N, 0, torch.zeros((2,1)))

dt = T/N

I_est = LQR_Ricatti.evaluate_J_X(Out_tt, Out_x, dt)

print(I_est.size())

'''
p = Out[0]

p_x = p[:,0].squeeze(1)
p_y = p[:,1].squeeze(1)

plt.plot(p_x,p_y)
plt.show()'''
