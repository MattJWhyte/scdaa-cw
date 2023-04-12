
from DGM import *
import torch
import lqr_solver
import matplotlib.pyplot as plt


# NB : Ricatti must be solved first
def get_samples(solver, num_samples):
    t = torch.rand(num_samples) * solver.T
    x = torch.rand(num_samples, 1, 2) * 6 - 3.0
    return t, x, solver.v(t, x)


# SET UP BASE LQR ----------------------------------------------
from base_solver import *
# Makes all matrices and lqr_solver available

# SOLVE RICATTI OVER GRID ------------------------------------

N = 10000
tt = np.linspace(0,T,N+1)
solver.solve_ricatti(tt)
solver.test_integrals()

print("Ricatti error", solver.test_Ricatti_ODE())

# ML TIME -------------------------------------------------

net = Net_DGM(dim_x=2, dim_S=100)

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=(10000,), gamma=0.1)
loss_fn = nn.MSELoss()

batch_size = 2048

running_loss = []

for it in range(100000):
    optimizer.zero_grad()

    train_t, train_x, train_v = get_samples(solver, batch_size)

    train_t = train_t.unsqueeze(1)
    train_x = train_x.squeeze(1)

    pred_v = net(train_t, train_x)
    loss = loss_fn(pred_v, train_v)

    if it % 10 == 0:
        print(torch.mean(train_v))
        print(torch.max(train_v))
        print("Epoch ", it, " loss ", loss)

    running_loss.append(loss)

    loss.backward()
    optimizer.step()
    scheduler.step()


plt.plot([i for i in range(len(running_loss))],[np.log(e) for e in running_loss])