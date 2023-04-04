
import torch
from helper import cuda, torchify_this
from torch.autograd import Variable

class DGM_Loss():
    def __init__(self, lqr_solver, u, a, te, xe, ye):
        self.lqr_solver = lqr_solver
        self.u = u
        self.a = a
        self.te = te
        self.xe = xe
        self.ye = ye

    def equation(self, x):

        d = torch.autograd.grad(self.u(x), x, grad_outputs=torch.ones_like(self.u(x)), create_graph=True)
        dt = d[0][:, 0].reshape(-1, 1)  # transform the vector into a column vector
        dx1 = d[0][:, 1].reshape(-1, 1)
        dx2 = d[0][:, 2].reshape(-1, 1)

        dx = torch.concat([dx1,dx2], dim=-1).unsqueeze(-1)

        # du/dxdx
        dx1x = torch.autograd.grad(dx1, x, grad_outputs=torch.ones_like(dx1), create_graph=True)[0][:, 1:].unsqueeze(-1)
        # du/dydy
        dx2x = torch.autograd.grad(dx2, x, grad_outputs=torch.ones_like(dx2), create_graph=True)[0][:, 1:].unsqueeze(-1)

        dxx = torch.cat([dx1x, dx2x], dim=-1)

        sigma = torchify_this(self.lqr_solver.sigma)
        H = torchify_this(self.lqr_solver.H)
        M = torchify_this(self.lqr_solver.M)
        C = torchify_this(self.lqr_solver.C)
        D = torchify_this(self.lqr_solver.D)

        a = sigma @ sigma.t() @ dxx
        tr = torch.sum(torch.diagonal(a, dim1=-1,dim2=-2), dim=-1).unsqueeze(-1)

        alpha = self.a(x).unsqueeze(-1)

        x_only = x[:,1:].unsqueeze(-1)

        c1 = (dx.transpose(-1,-2) @ H @ x_only).squeeze(-1)
        c2 = (dx.transpose(-1,-2) @ M @ alpha).squeeze(-1)
        c3 = (x_only.transpose(-1,-2) @ C @ x_only).squeeze(-1)
        c4 = alpha.transpose(-1,-2) @ D @ alpha

        f = dt + 0.5*tr + c1 + c2 + c3 + c4
        return f**2

    def terminal(self, x):
        x_only = x[:,1:].unsqueeze(-1)
        R = torchify_this(self.lqr_solver.R)
        L = self.u(x) - (x_only.transpose(-1,-2) @ R @ x_only).squeeze(-1)
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