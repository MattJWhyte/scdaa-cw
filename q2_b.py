import matplotlib.pyplot as plt

from FFN import *
import torch
import lqr_solver
import numpy as np
import matplotlib.pyplot as plt

from helper import positive_def_matrix

# NB : Ricatti must be solved first
def get_a_samples(solver, num_samples):
    t = torch.rand(num_samples) * solver.T
    x = torch.rand(num_samples, 1, 2) * 6 - 3.0
    return t, x, solver.a(t, x)

# SET UP LQR ----------------------------------------------

H = positive_def_matrix()
M = positive_def_matrix() #np.array([[0.15663973 0.15513884],[0.15513884 0.20362521]])

sigma = positive_def_matrix()
T = 1
C = positive_def_matrix()
D = positive_def_matrix()
R = positive_def_matrix()

lqr_s = lqr_solver.LQR_Solver(H, M, sigma, T, C, D, R)

# SOLVE RICATTI OVER GRID ------------------------------------

N = 10000
tt = np.linspace(0,T,N+1)
lqr_s.solve_ricatti(tt)
lqr_s.test_integrals()

print("Ricatti error", lqr_s.test_Ricatti_ODE())

# ML TIME -------------------------------------------------

net = FFN(sizes=[3, 100, 100, 2])

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=(10000,), gamma=0.1)
loss_fn = nn.MSELoss()

batch_size = 2048

loss_track = []

num_epochs = 10000

for it in range(num_epochs):
    optimizer.zero_grad()

    train_t, train_x, train_a = get_a_samples(lqr_s, batch_size)

    train_t = train_t.unsqueeze(1)
    train_x = train_x.squeeze(1)

    train_tx = torch.cat([train_t, train_x], 1)
    pred_a = net(train_tx)
    loss = loss_fn(pred_a, train_a)

    loss_track.append(loss.detach().numpy())

    if it % 10 == 0:
        print(torch.mean(torch.abs(train_a)))
        print(torch.mean(torch.abs(pred_a)))
        print("Epoch ", it, " loss ", loss)

    loss.backward()
    optimizer.step()
    scheduler.step()

plt.clf()

plt.plot([i for i in range(num_epochs)], loss_track)
plt.savefig("loss.png")

