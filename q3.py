
import torch
from torch.autograd import Variable
import torch.nn as nn
from helper import positive_def_matrix, torchify_this, cuda
import matplotlib.pyplot as plt
import numpy as np
from DGM import *
from lqr_solver import LQR_Solver


H = np.eye(2)
M = np.eye(2)

sigma = 0.05*np.eye(2)
T = 1.0
C = 0.1*np.eye(2)
D = 0.1*np.eye(2)
R = np.eye(2)

lqr_solver = LQR_Solver(H, M, sigma, T, C, D, R)
lqr_solver.solve_ricatti(np.linspace(0,T,5001))

batch_size = 25
x_ini = torch.rand(batch_size, 1, 2) * 2 - 1.0
t_ini = torch.rand(batch_size)*T

v_ini_mc = torch.zeros(batch_size)

x_ini = x_ini.transpose(-1, -2)

N = 5000

i = 0
for t,x in zip(t_ini,x_ini[:]):
    out_t, out_x = lqr_solver.simulate_X(5000, N, t, x)
    J_est = lqr_solver.evaluate_J_X(out_t, out_x, T / N)
    J_est_mean = torch.mean(J_est)
    v_ini_mc[i] = J_est_mean
    i += 1
    print("sim {}".format(i))


H = torchify_this(H)
M = torchify_this(M)
sigma = torchify_this(sigma)
C = torchify_this(C)
D = torchify_this(D)
R = torchify_this(R)


def get_gradient(output, x):
    grad = torch.autograd.grad(output, x, grad_outputs=torch.ones_like(output), create_graph=True, retain_graph=True, only_inputs=True)[0]
    return grad


class Net(nn.Module):
    # n_layer: the number of hidden layers
    # n_hidden: the number of vertices in each layer
    def __init__(self, n_layer, n_hidden, dim):
        super(Net, self).__init__()
        self.dim = dim
        self.input_layer = nn.Linear(dim, n_hidden)
        self.hidden_layers = nn.ModuleList([nn.Linear(n_hidden, n_hidden) for i in range(n_layer)])
        self.output_layer = nn.Linear(n_hidden, 1)

    def forward(self, x):
        o = self.act(self.input_layer(x))

        for i, li in enumerate(self.hidden_layers):
            o = self.act(li(o))

        out = self.output_layer(o)

        return out

    def act(self, x):
        return x * torch.sigmoid(x)
        # return torch.sigmoid(x)
        # return torch.tanh(x)
        # return torch.relu(x)


class PDE():
    def __init__(self, net, te, xe, ye):
        self.net = net
        self.te = te
        self.xe = xe
        self.ye = ye

    def equation(self, x):

        d = torch.autograd.grad(self.net(x), x, grad_outputs=torch.ones_like(self.net(x)), create_graph=True)
        dt = d[0][:, 0].reshape(-1, 1)  # transform the vector into a column vector
        dx1 = d[0][:, 1].reshape(-1, 1)
        dx2 = d[0][:, 2].reshape(-1, 1)

        dx = torch.concat([dx1,dx2], dim=-1).unsqueeze(-1)

        # du/dxdx
        dx1x = torch.autograd.grad(dx1, x, grad_outputs=torch.ones_like(dx1), create_graph=True)[0][:, 1:].unsqueeze(-1)

        # du/dydy
        dx2x = torch.autograd.grad(dx2, x, grad_outputs=torch.ones_like(dx2), create_graph=True)[0][:, 1:].unsqueeze(-1)

        dxx = torch.cat([dx1x, dx2x], dim=-1)

        a = sigma @ sigma.t() @ dxx
        tr = torch.sum(torch.diagonal(a, dim1=-1,dim2=-2), dim=-1).unsqueeze(-1)

        alpha = cuda(torch.ones((2,1)))

        x_only = x[:,1:].unsqueeze(-1)

        c1 = (dx.transpose(-1,-2) @ H @ x_only).squeeze(-1)
        c2 = (dx.transpose(-1,-2) @ M @ alpha).squeeze(-1)
        c3 = (x_only.transpose(-1,-2) @ C @ x_only).squeeze(-1)
        c4 = alpha.t() @ D @ alpha

        f = dt + 0.5*tr + c1 + c2 + c3 + c4
        return f**2

    def terminal(self, x):
        x_only = x[:,1:].unsqueeze(-1)
        L = self.net(x) - (x_only.transpose(-1,-2) @ R @ x_only).squeeze(-1)
        return L**2

    def loss_func(self, size=2 ** 8):
        x = torch.cat(
            (torch.rand([size, 1]) * self.te, torch.rand([size, 1]) * self.xe * 2 - self.xe, torch.rand([size, 1]) * self.ye * 2 - self.ye), dim=1)
        x = cuda(x)
        x = Variable(x, requires_grad=True)

        x_T = torch.cat([cuda(torch.ones(size,1)), x[:,1:]], dim=-1)

        diff_error = self.equation(x)
        terminal_error = self.terminal(x_T)

        return torch.mean(diff_error + terminal_error), torch.mean(diff_error), torch.mean(terminal_error)

net = Net(3, 100, dim=3)
net_DGM = cuda(Net_DGM(dim_x=2, dim_S=100))


num_epochs = 5000

optimizer = torch.optim.Adam(net_DGM.parameters(), lr=0.01)
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=(1000,2000,3000,4000,5000,), gamma=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.75)
loss_fn = nn.MSELoss()

batch_size = 10000

loss_track = []
diff_e_track = []
terminal_e_track = []

error_track = []

def net_f(x):
    return net_DGM(x[:,0].unsqueeze(-1), x[:,1:])

pde_loss = PDE(net_f, T, 1.0, 1.0)

for it in range(num_epochs+1):
    optimizer.zero_grad()

    loss, diff_e, terminal_e = pde_loss.loss_func(batch_size)

    if True:
        loss_track.append(loss.cpu().detach().numpy())
        diff_e_track.append(diff_e.cpu().detach().numpy())
        terminal_e_track.append(terminal_e.cpu().detach().numpy())

    if it % 10 == 0:
        print("Epoch ", it, " loss ", loss, " lr ", scheduler.get_last_lr())

    if it % 100 == 0:
        plt.clf()
        plt.plot([j for j in range(len(loss_track))], np.log(loss_track), label='loss')
        plt.plot([j for j in range(len(loss_track))], np.log(diff_e_track), label='equation loss')
        plt.plot([j for j in range(len(loss_track))], np.log(terminal_e_track), label='terminal loss')
        plt.legend()
        plt.title("Train log-loss for DGM")
        plt.ylabel("log-loss")
        plt.xlabel("number of epoch")
        plt.savefig("q3-loss.png")

    if it % 1000 == 0 and it > 0:
        pred_v = net_DGM(cuda(t_ini.unsqueeze(-1)), cuda(x_ini.squeeze(-1)))
        print(nn.MSELoss()(cuda(v_ini_mc), pred_v.squeeze(-1)))
        error_track.append(float(nn.MSELoss()(cuda(v_ini_mc), pred_v.squeeze(-1)).cpu().detach().numpy()))
        plt.clf()
        plt.plot([j*1000 for j in range(len(error_track))], error_track)
        plt.title("MSE of DGM against MC Estimate")
        plt.ylabel("Mean-Square Error")
        plt.xlabel("number of epoch")
        plt.savefig("q3-dgm-mc.png")

    loss.backward()
    optimizer.step()
    scheduler.step()

'''
plt.clf()
plt.plot([j for j in range(len(loss_track))], loss_track, label='loss')
plt.plot([j for j in range(len(loss_track))], diff_e_track, label='diff_e')
plt.plot([j for j in range(len(loss_track))], terminal_e_track, label='terminal_e')
plt.legend()
plt.savefig("q3-loss.png")'''


print("-----------------")
print("Evaluating models:")

t = torch.rand(1000) * lqr_solver.T
x = torch.rand(1000, 1, 2) * 1.0 * 2.0 - 1.0

v_true = cuda(lqr_solver.v(t, x))

t = cuda(t.unsqueeze(1))
x = cuda(x.squeeze(1))

pred_v = net_DGM(t, x)
v_loss = nn.MSELoss()(v_true, pred_v)

i = torch.argmax((pred_v-v_true)**2)

plt.plot([i for i in range(1000)], ((pred_v-v_true)**2).cpu().numpy())
plt.savefig("testloss.png")

print(pred_v[i], v_true[i])
print("v loss: ", v_loss)
