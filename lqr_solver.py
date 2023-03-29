
import torch
import numpy as np
from scipy.integrate import odeint

class LQR_Solver:

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

    def solve_ricatti(self, tt):
        def S_prime(S,t):
            # BIG PROBLEM
            S = np.reshape(S,(2,2))
            return (-2*self.H.T @ S + S @ self.M @ np.linalg.inv(self.D) @ self.M @ S - self.C).flatten()

        # Solves S ′ (r) = −2 H^T S(r) + S_t M D^{-1} M S(r) − C , r in [t,T] , S(T) = R
        return np.reshape(odeint(S_prime, self.R.flatten(), tt[::-1])[::-1],(tt.shape[0],2,2))

    def evaluate_integral(self, S, tt):
        def I_prime(I, t):
            idx = (np.abs(tt - t)).argmin()
            return -np.trace(self.sigma @ self.sigma.T @ S[idx])
        I_t = odeint(I_prime, 0, tt[::-1])
        return I_t


    # t.size = [batch_size]
    # x.size = [batch_size, 1, 2]
    def v(self, tt, xx):
        S = self.solve_ricatti(tt.numpy())
        # I_t numpy array with shape [batch_size x 1]
        I_t = self.evaluate_integral(S, tt.numpy())
        I_t = torch.from_numpy(I_t).float()
        S = torch.from_numpy(S.copy()).float()

        L = torch.bmm(xx,S)
        L_2 = torch.bmm(L, xx.transpose(1,2)).squeeze(1)
        O_1 = L_2 + I_t
        O_2 = (xx@S@xx.transpose(1,2)).squeeze(1) + I_t

        assert torch.equal(O_1,O_2)

        return O_1

    # t.size = [batch_size]
    # x.size = [batch_size, 1, 2]
    def a(self, tt, xx): # How do we remove S_T??
        # a(t, x) = −DM ⊤ S(t)x
        S = self.solve_ricatti(tt.numpy())
        S = torch.from_numpy(S.copy()).float()
        K = torch.from_numpy(-self.D@self.M.transpose()).float()
        return (K@S@xx.transpose(1,2)).squeeze(-1) #torch.bmm(torch.bmm(K, S), xx.transpose(1,2))


    def simulate_X(self, N, n_realisations, t_init, x_init):
        dt = self.T/N
        tt = np.linspace(0,self.T,N+1)
        S = self.solve_ricatti(tt)
        S = torch.from_numpy(S.copy()).float()

        k_init = int(t_init/dt)
        N_prime = N-k_init

        tt = torch.from_numpy(tt).float()[k_init:]

        dW = (dt)**(0.5)*torch.randn((n_realisations, N_prime, 2, 1))

        X = torch.zeros((n_realisations, N_prime+1, 2, 1))

        X[:,0,:,:] = x_init

        for i in range(N_prime):
            X[:,i+1,:,:] = X[:,i,:,:] + dt*(torch.from_numpy(self.H).float() @ X[:,i,:,:] - torch.from_numpy(self.M).float() @ torch.from_numpy(self.D).float() @ torch.from_numpy(self.M).float().transpose(0,1) @ S[i+k_init] @ X[:,i,:,:])
            X[:,i+1,:,:] += torch.from_numpy(self.sigma).float() @ dW[:,i,:,:]

        return tt,X

    def evaluate_J_X(self, tt, X, dt):
        a = torch.zeros(X.size())
        for j in range(X.size()[0]):
            a[j] = self.a(tt, X[j].transpose(1,2)).unsqueeze(-1)

        I = X.transpose(2,3) @ torch.from_numpy(self.C).float() @ X
        I += a.transpose(2,3) @ torch.from_numpy(self.D).float() @ a
        I = dt * torch.sum(I, dim=1) + X[:,-1,:,:].transpose(1,2) @ torch.from_numpy(self.R).float() @ X[:,-1,:,:]

        return I.squeeze(1).squeeze(1)


