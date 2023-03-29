
import torch


class LQR_Ricatti_Solver:

    def __init__(self, H, M, sigma, x_init, T, C, D, R):

        assert T > 0
        # Maybe assert C,R >= 0 and D > 0
        self.H = H
        self.M = M
        self.sigma = sigma
        self.x_init = x_init
        self.T = T
        self.C = C
        self.D = D
        self.R = R

    def solve_ricatti(self, tt, S_T):
        # Solves S ′ (r) = −2 H^T S(r) + S_t M D^{-1} M S(r) − C , r in [t,T] , S(T) = R
        return torch.zeros([tt.size[0], 2, 2])

    # t.size = [batch_size]
    # x.size = [batch_size, 1, 2]
    def v(self, t, x):
        pass

    # t.size = [batch_size]
    # x.size = [batch_size, 1, 2]
    def a(self, tt, xx, S_T): # How do we remove S_T??
        # a(t, x) = −DM ⊤ S(t)x
        S = self.solve_ricatti(tt, S_T)
        K = -self.D@self.M.transpose()
        return torch.bmm(torch.bmm(K, S), xx.transpose(1, 2))

