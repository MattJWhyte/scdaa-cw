

import torch
from helper import cuda
from torch.autograd import Variable

class Hamiltonian():
    def __init__(self, lqr_solver, v, a, te, xe, ye):
        self.lqr_solver = lqr_solver
        self.v = v
        self.a = a
        self.te = te
        self.xe = xe
        self.ye = ye

    def equation(self, tx):

        M = cuda(torch.from_numpy(self.lqr_solver.M).float())
        H = cuda(torch.from_numpy(self.lqr_solver.H).float())
        C = cuda(torch.from_numpy(self.lqr_solver.C).float())
        D = cuda(torch.from_numpy(self.lqr_solver.D).float())

        d = torch.autograd.grad(self.v(tx), tx, grad_outputs=torch.ones_like(self.v(tx)), create_graph=True)
        dv_dx1 = d[0][:, 1].reshape(-1, 1)
        dv_dx2 = d[0][:, 2].reshape(-1, 1)
        dv_dx = torch.concat([dv_dx1,dv_dx2], dim=-1).unsqueeze(-1)

        x = tx[:,1:].unsqueeze(-1)

        c1 = (dv_dx.transpose(-1,-2) @ H @ x).squeeze(-1)
        c2 = (dv_dx.transpose(-1,-2) @ M @ self.a(tx).unsqueeze(-1)).squeeze(-1)
        c3 = (x.transpose(-1,-2) @ C @ x).squeeze(-1)
        c4 = ((self.a(tx) @ D).unsqueeze(-2) @ self.a(tx).unsqueeze(-1)).squeeze(-1)

        return c1 + c2 + c3 + c4

    def loss_func(self, size=2 ** 8):
        tx = torch.cat(
            (torch.rand([size, 1]) * self.te, torch.rand([size, 1]) * self.xe * 2 - self.xe, torch.rand([size, 1]) * self.ye * 2 - self.ye), dim=1)
        tx = cuda(tx)
        tx = Variable(tx, requires_grad=True)
        return torch.mean(self.equation(tx))