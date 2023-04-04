

import torch
from FFN import FFN
from DGM import Net_DGM
from helper import cuda
from DGM_loss import DGM_Loss
from helper import *
from lqr_solver import LQR_Solver
from Hamiltonian import Hamiltonian



H = positive_def_matrix()
M = positive_def_matrix() #np.array([[0.15663973 0.15513884],[0.15513884 0.20362521]])

sigma = positive_def_matrix()
T = 1
C = positive_def_matrix()
D = positive_def_matrix()
R = positive_def_matrix()

lqr_solver = LQR_Solver(H, M, sigma, T, C, D, R)

v_net = cuda(Net_DGM(dim_x=2, dim_S=100))
a_net = cuda(FFN(sizes=[3, 100, 100, 2]))

def v_net_f(x):
    return v_net(x[:,0].unsqueeze(-1), x[:,1:])

v_loss = DGM_Loss(lqr_solver, v_net_f, a_net, T, 5.0, 5.0)
a_loss = Hamiltonian(lqr_solver, v_net_f, a_net, T, 5.0, 5.0)

num_iter = 100

for i in range(num_iter):
    print("------------\nV")
    v_net = cuda(Net_DGM(dim_x=2, dim_S=100))
    def v_net_f(x):
        return v_net(x[:, 0].unsqueeze(-1), x[:, 1:])
    v_loss = DGM_Loss(lqr_solver, v_net_f, a_net, T, 5.0, 5.0)
    train_net(v_net, v_loss, 5000, 5000, threshold=1.0)

    print("------------\nA")
    a_net = cuda(FFN(sizes=[3, 100, 100, 2]))
    a_loss = Hamiltonian(lqr_solver, v_net_f, a_net, T, 5.0, 5.0)
    train_net(a_net, a_loss, 10000, 10000)


