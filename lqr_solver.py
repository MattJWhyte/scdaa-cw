import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.integrate import odeint

class LQR_Solver:

    def __init__(self, H, M, sigma, T, C, D, R):

        assert T > 0
        # Maybe assert C,R >= 0 and D > 0

        self.H = H
        self.M = M
        self.sigma = sigma
        self.T = T
        self.C = C
        self.D = D
        self.R = R

        # S(t)
        self.S = None
        self.time_grid = None
        self.dt = None
        self.I = None

    def solve_ricatti(self, tt):
        def S_prime(S,t):
            # BIG PROBLEM
            S = np.reshape(S,(2,2))
            return (-2*self.H.T @ S + S @ self.M @ np.linalg.inv(self.D) @ self.M @ S - self.C).flatten()

        # Solves S ′ (r) = −2 H^T S(r) + S_t M D^{-1} M S(r) − C , r in [t,T] , S(T) = R
        self.S = np.reshape(odeint(S_prime, self.R.flatten(), tt[::-1])[::-1],(tt.shape[0],2,2))
        self.S = torch.from_numpy(self.S.copy()).float()
        self.time_grid = tt
        self.dt = self.time_grid[1]-self.time_grid[0]
        self.I = self.evaluate_integral_2().unsqueeze(-1)
        return self.S

    def test_Ricatti_ODE(self):

        plt.clf()
        plt.plot(self.time_grid,self.S[:,0,0], label='0,0')
        plt.plot(self.time_grid, self.S[:, 0, 1], label='0,1')
        plt.plot(self.time_grid, self.S[:, 1, 0], label='1,0')
        plt.plot(self.time_grid, self.S[:, 1, 1], label='1,1')
        plt.legend()
        plt.savefig("S_t")
        def S_prime(S):
            return -2.0 * (self.H.T @ S) + S @ self.M @ np.linalg.inv(self.D) @ self.M @ S - self.C

        St_deriv = S_prime(self.S.numpy())

        e_l = 0.0
        e_r = 0.0

        N = self.S.size()[0]-1

        for i in range(N):
            d_prime = (self.S[i + 1].numpy() - self.S[i].numpy()) / self.dt

            e_l += np.abs(St_deriv[i] - d_prime) / N
            e_r += np.abs(St_deriv[i + 1] - d_prime) / N

        return max(np.max(e_l), np.max(e_r))

    def dummy_S(self,tt):
        self.S = torch.zeros((tt.shape[0],2,2))
        self.S[:,0,0] = torch.from_numpy(tt).float()
        self.I = self.evaluate_integral_2()

    def evaluate_integral(self):
        def I_prime(I, t):
            idx = (np.abs(self.time_grid - t)).argmin()
            return -np.trace(self.sigma @ self.sigma.T @ self.S[idx].numpy())
        I_t = odeint(I_prime, 0, self.time_grid[::-1])
        return I_t[::-1]

    def test_integrals(self):
        I_t = self.evaluate_integral()
        I_t_2 = self.evaluate_integral_2()
        plt.plot(self.time_grid, I_t_2.numpy())
        plt.plot(self.time_grid, I_t)
        I_t = torch.from_numpy(I_t.copy()).float().squeeze(-1)
        print("I sanity check")
        print("All close", False not in torch.isclose(I_t_2, I_t))
        print(I_t_2.size())
        print(I_t.size())
        print("Max diff", torch.max(torch.abs(I_t_2-I_t)))
        print("Max diff idx", torch.argmax(torch.abs(I_t_2 - I_t)))
        plt.savefig("i_t_comparison.png")

    def evaluate_integral_2(self):
        sigma = torch.from_numpy(self.sigma).float()
        integrand = torch.zeros((self.S.size(0)))
        for i in range(integrand.size(0)):
            integrand[i] = torch.trace(sigma @ sigma.t() @ self.S[i])
        integral = torch.flip(torch.cumsum(torch.flip(integrand,dims=[0]), dim=0), dims=[0])*self.dt
        return integral

    def at(self, xx, tt):
        kk = (tt / self.dt).type(torch.int64)
        S = torch.index_select(xx, dim=0, index=kk)
        return S

    # t.size = [batch_size]
    # x.size = [batch_size, 1, 2]
    def v(self, tt, xx):
        #S = self.solve_ricatti(tt.numpy()) # TODO NB: Using pre-computed S
        # I_t numpy array with shape [batch_size x 1]
        #I_t = self.evaluate_integral() # TODO NB: Removed dependency on S,tt
        #I_t = torch.from_numpy(I_t.copy()).float()
        ##S = torch.from_numpy(S.copy()).float()

        I_t = self.I

        L = torch.bmm(xx,self.at(self.S, tt))
        L_2 = torch.bmm(L, xx.transpose(1,2)).squeeze(1)
        O_1 = L_2 + self.at(I_t,tt)
        O_2 = (xx@self.at(self.S, tt)@xx.transpose(1,2)).squeeze(1) + self.at(I_t,tt)

        assert torch.equal(O_1,O_2)

        return O_1

    # t.size = [batch_size]
    # x.size = [batch_size, 1, 2]
    def a(self, tt, xx): # How do we remove S_T??
        # a(t, x) = −DM ⊤ S(t)x
        #S = self.solve_ricatti(tt.numpy())
        #S = torch.from_numpy(S.copy()).float()
        K = torch.from_numpy(-np.linalg(self.D)@self.M.transpose()).float()

        return (K@self.at(self.S, tt)@xx.transpose(-1,-2)).squeeze(-1) #torch.bmm(torch.bmm(K, S), xx.transpose(1,2))


    def simulate_X(self, N, n_realisations, t_init, x_init):
        dt = self.T/N

        assert dt == self.dt

        tt = np.linspace(0,self.T,N+1)
        #S = self.solve_ricatti(tt)
        #S = torch.from_numpy(S.copy()).float()

        k_init = int(t_init/dt)
        N_prime = N-k_init

        tt = torch.from_numpy(tt).float()[k_init:]

        dW = (dt)**(0.5)*torch.randn((n_realisations, N_prime, 2, 1))

        X = torch.zeros((n_realisations, N_prime + 1, 2, 1))
        X[:, 0, :, :] = x_init

        for i in range(N_prime):
            M = torch.from_numpy(self.M).float()
            H = torch.from_numpy(self.H).float()
            D = torch.from_numpy(self.D).float()
            sigma = torch.from_numpy(self.sigma).float()

            X[:, i+1, :, :] = X[:, i, :, :] + dt*(H @ X[:, i, :, :] - M @ np.linalg(D) @ M.transpose(0,1) @ self.S[k_init+i] @ X[:,i,:,:])
            X[:, i+1, :, :] += sigma @ dW[:,i,:,:]

        return tt,X


    def evaluate_J_X(self, tt, X, dt):

        assert self.dt == dt

        a = self.a(tt, X.transpose(-1,-2)).unsqueeze(-1)

        terminal = X[:,-1,:,:].transpose(1,2) @ torch.from_numpy(self.R).float() @ X[:,-1,:,:]

        I = X.transpose(-1,-2) @ torch.from_numpy(self.C).float() @ X
        I += a.transpose(-1,-2) @ torch.from_numpy(self.D).float() @ a

        I = dt * torch.sum(I, dim=1) + terminal

        print(terminal[0,-1])
        print(I.size())
        print(terminal.size())

        return I.squeeze(1).squeeze(1)


