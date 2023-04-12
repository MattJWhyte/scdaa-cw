

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

    def get_gradient(self, output, x):
        grad = \
        torch.autograd.grad(output, x, grad_outputs=torch.ones_like(output), create_graph=True, retain_graph=True,
                            only_inputs=True)[0]
        return grad

    def equation(self, tx):
        '''
        for param in self.v_net.parameters():
            param.requires_grad = False'''

        M = cuda(torch.from_numpy(self.lqr_solver.M).float())
        H = cuda(torch.from_numpy(self.lqr_solver.H).float())
        C = cuda(torch.from_numpy(self.lqr_solver.C).float())
        D = cuda(torch.from_numpy(self.lqr_solver.D).float())

        d = torch.autograd.grad(self.v(tx), tx, grad_outputs=torch.ones_like(self.v(tx)))[0].detach()

        #print("d size: ", d.size())
        dv_dx1 = d[:, 1].reshape(-1, 1)
       # print("dv dx size: ", dv_dx1.size())
        dv_dx2 = d[:, 2].reshape(-1, 1)
        dv_dx = torch.concat([dv_dx1,dv_dx2], dim=-1).unsqueeze(-1)

        x = tx[:,1:].unsqueeze(-1)

        alpha = self.a(tx).unsqueeze(-1)

        dv_dx_T = cuda(torch.from_numpy(dv_dx.transpose(-1,-2).cpu().numpy()))

        c1 = (dv_dx_T @ H @ x).squeeze(-1)
        c2 = (dv_dx_T @ M @ alpha).squeeze(-1)
        c3 = (x.transpose(-1,-2) @ C @ x).squeeze(-1)
        c4 = (alpha.transpose(-1,-2) @ D @ alpha).squeeze(-1)

        return [c1 + c2 + c3 + c4, c1, c2, c3, c4]

    def loss_func(self, size=2 ** 8):
        tx = torch.cat(
            (torch.rand([size, 1]) * self.te, torch.rand([size, 1]) * self.xe * 2 - self.xe, torch.rand([size, 1]) * self.ye * 2 - self.ye), dim=1)
        tx = Variable(tx, requires_grad=True)
        tx = cuda(tx)
        return [torch.mean(f) for f in self.equation(tx)]



class Hamiltonian2():
    def __init__(self, lqr_solver, v_net, v, a, te, xe, ye):
        self.v_net = v_net
        self.lqr_solver = lqr_solver
        self.v = v
        self.a = a
        self.te = te
        self.xe = xe
        self.ye = ye

    def equation(self, tx):
        '''
        for param in self.v_net.parameters():
            param.requires_grad = False'''

        M = cuda(torch.from_numpy(self.lqr_solver.M).float())
        H = cuda(torch.from_numpy(self.lqr_solver.H).float())
        C = cuda(torch.from_numpy(self.lqr_solver.C).float())
        D = cuda(torch.from_numpy(self.lqr_solver.D).float())

        d = torch.autograd.grad(self.v(tx), tx, grad_outputs=torch.ones_like(self.v(tx)))[0].detach()
        dv_dx1 = d[:, 1].reshape(-1, 1)
        dv_dx2 = d[:, 2].reshape(-1, 1)
        dv_dx = torch.concat([dv_dx1,dv_dx2], dim=-1).unsqueeze(-1)

        x = tx[:,1:].unsqueeze(-1)

        alpha = self.a(tx).unsqueeze(-1)

        c1 = (dv_dx.transpose(-1,-2) @ H @ x).squeeze(-1)
        c2 = (dv_dx.transpose(-1,-2) @ M @ alpha).squeeze(-1)
        c3 = (x.transpose(-1,-2) @ C @ x).squeeze(-1)
        c4 = (alpha.transpose(-1,-2) @ D @ alpha).squeeze(-1)

        return [c1 + c2 + c3 + c4, c1, c2, c3, c4]

    def loss_func(self, size=2 ** 8):
        tx = torch.cat(
            (torch.rand([size, 1]) * self.te, torch.rand([size, 1]) * self.xe * 2 - self.xe, torch.rand([size, 1]) * self.ye * 2 - self.ye), dim=1)
        tx = cuda(tx)
        tx = Variable(tx, requires_grad=True)
        return [-0.0*torch.mean(f) for f in self.equation(tx)]