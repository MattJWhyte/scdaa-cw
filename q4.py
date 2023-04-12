

import torch
from FFN import FFN
from DGM import Net_DGM
from helper import cuda
from DGM_loss import DGM_Loss
from helper import *
from lqr_solver import LQR_Solver
from Hamiltonian import Hamiltonian
from torch import nn
import matplotlib.pyplot as plt


H = np.eye(2)
M = np.eye(2)

sigma = 0.05*np.eye(2)
T = 1
C = 0.1*np.eye(2)
D = 0.1*np.eye(2)
R = np.eye(2)

lqr_solver = LQR_Solver(H, M, sigma, T, C, D, R)
lqr_solver.solve_ricatti(np.linspace(0,T,5000))

v_net = cuda(Net_DGM(dim_x=2, dim_S=100))
a_net = cuda(FFN(sizes=[3, 100, 100, 2]))

def v_net_f(x):
    return v_net(x[:,0].unsqueeze(-1), x[:,1:])

v_loss = DGM_Loss(lqr_solver, v_net_f, a_net, T, 1.0, 1.0)
a_loss = Hamiltonian(lqr_solver, v_net_f, a_net, T, 1.0, 1.0)

num_iter = 500

running_a_loss = []
running_v_loss = []

for i in range(num_iter):

    print("-----------------")
    print("Evaluating models:")

    t = torch.rand(1000) * lqr_solver.T
    x = torch.rand(1000, 1, 2) * 6 - 3.0

    v_true = cuda(lqr_solver.v(t, x))
    a_true = cuda(lqr_solver.a(t, x))

    t = cuda(t.unsqueeze(1))
    x = cuda(x.squeeze(1))

    pred_v = v_net(t, x)
    v_loss = nn.MSELoss()(v_true, pred_v)
    print("v loss: ", v_loss)
    running_v_loss.append(v_loss.cpu().detach().numpy())

    pred_a = a_net(torch.cat([t,x], dim=1))
    a_loss = nn.MSELoss()(a_true, pred_a)
    print("a loss: ", a_loss)
    running_a_loss.append(a_loss.cpu().detach().numpy())
    print("\n")

    plt.plot([j for j in range(len(running_a_loss))], running_a_loss)
    plt.xlabel("Iteration")
    plt.ylabel("MSE against true a(t,x)")
    plt.savefig("q4/a-loss-iter-{}.png".format(i))

    plt.clf()

    plt.plot([j for j in range(len(running_v_loss))], running_v_loss)
    plt.xlabel("Iteration")
    plt.ylabel("MSE against true v(t,x)")
    plt.savefig("q4/v-loss-iter-{}.png".format(i))

    plt.clf()

    print("------------\nV")
    v_net = cuda(Net_DGM(dim_x=2, dim_S=100))
    def v_net_f(x):
        return v_net(x[:, 0].unsqueeze(-1), x[:, 1:])
    v_loss = DGM_Loss(lqr_solver, v_net_f, a_net, T, 1.0, 1.0)
    train_net(v_net, v_loss, 10000, 5000, lr=0.01, threshold=0.00001)

    print("------------\nA")
    a_net = cuda(FFN(sizes=[3, 100, 100, 2]))
    a_loss = Hamiltonian(lqr_solver, v_net_f, a_net, T, 1.0, 1.0)
    train_net(a_net, a_loss, 10000, 5000, lr=0.005)






