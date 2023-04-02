
from DGM import *
import torch
import lqr_solver
from helper import positive_def_matrix

# NB : Ricatti must be solved first
def get_samples(solver, num_samples):
    t = torch.rand(num_samples) * solver.T
    x = torch.rand(num_samples, 1, 2) * 6 - 3.0
    return t, x, solver.v(t, x)

# SET UP LQR ----------------------------------------------

H = np.eye(2) #positive_def_matrix()
M = np.eye(2) #positive_def_matrix() #np.array([[0.15663973 0.15513884],[0.15513884 0.20362521]])

sigma = np.array([[0.05, 0.0],[0.05,0.0]])#0.05*np.eye(2) #positive_def_matrix()
T = 1
C = 0.1 * np.eye(2) #positive_def_matrix()
D = 0.1 * np.eye(2) #positive_def_matrix()
R = np.eye(2) #positive_def_matrix()

lqr_s = lqr_solver.LQR_Solver(H, M, sigma, T, C, D, R)

# SOLVE RICATTI OVER GRID ------------------------------------

N = 10000
tt = np.linspace(0,T,N+1)
lqr_s.solve_ricatti(tt)
lqr_s.test_integrals()

print("Ricatti error", lqr_s.test_Ricatti_ODE())

# ML TIME -------------------------------------------------

net = Net_DGM(dim_x=2, dim_S=100)

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=(10000,), gamma=0.1)
loss_fn = nn.MSELoss()

batch_size = 2048

for it in range(100000):
    optimizer.zero_grad()

    train_t, train_x, train_v = get_samples(lqr_s, batch_size)

    train_t = train_t.unsqueeze(1)
    train_x = train_x.squeeze(1)

    pred_v = net(train_t, train_x)
    loss = loss_fn(pred_v, train_v)

    if it % 10 == 0:
        print(torch.mean(train_v))
        print(torch.max(train_v))
        print("Epoch ", it, " loss ", loss)

    loss.backward()
    optimizer.step()
    scheduler.step()
