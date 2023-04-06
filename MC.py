#!/usr/bin/env python
# coding: utf-8

# In[130]:


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
        K = torch.from_numpy(-np.linalg.inv(self.D)@self.M.transpose()).float()

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

            X[:, i+1, :, :] = X[:, i, :, :] + dt*(H @ X[:, i, :, :] - M @ np.linalg.inv(D) @ M.transpose(0,1) @ self.S[k_init+i] @ X[:,i,:,:])
            X[:, i+1, :, :] += sigma @ dW[:,i,:,:]

        return tt,X


    def evaluate_J_X(self, tt, X, dt):

        assert self.dt == dt

        a = self.a(tt, X.transpose(-1,-2)).unsqueeze(-1)
        

        terminal = X[:,-1,:,:].transpose(1,2) @ torch.from_numpy(self.R).float() @ X[:,-1,:,:]

        I = X.transpose(-1,-2) @ torch.from_numpy(self.C).float() @ X
        I += a.transpose(-1,-2) @ torch.from_numpy(self.D).float() @ a

        I = dt * torch.sum(I, dim=1) + terminal

      

        return I.squeeze(1).squeeze(1)


# In[235]:


from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import torch
from helper import positive_def_matrix

'''
T = 5
t = np.linspace(T,0,100)

y0 = np.eye(2).flatten()

def der(y,t):
    return np.eye(2).flatten()

S = np.reshape(odeint(der, y0, t)[::-1],(100,2,2))

print(S[0])

print(S[-1])'''

torch.manual_seed(0)

# SET UP LQR ----------------------------------------------

H = np.eye(2) #positive_def_matrix()
M = np.eye(2) #positive_def_matrix() #np.array([[0.15663973 0.15513884],[0.15513884 0.20362521]])

sigma = 0.05*np.eye(2) #np.array([[0.05, 0.0],[0.05,0.0]])#0.05*np.eye(2) #positive_def_matrix()
T = 1
C = 0.1*np.eye(2) #positive_def_matrix()
D = 0.1*np.eye(2) #positive_def_matrix()
R = np.eye(2) #positive_def_matrix()
T = 1.0


LQR_Ricatti = LQR_Solver(H, M, sigma, T, C, D, R)


N = 1000
tt = np.linspace(0,T,N+1)

St = LQR_Ricatti.solve_ricatti(tt).numpy()

num_realisation = 100000
timesteps = [1, 10, 50, 100, 500, 1000]
MSE_step = []
for N in timesteps:
    tt = np.linspace(0,T,N+1)
    St = LQR_Ricatti.solve_ricatti(tt).numpy()
    dt = T/N
    MSE_t = []
    i = 0
    for x in ini_x[:]:
        i+=1
        print(i)
        out_t, out_x = LQR_Ricatti.simulate_X(N, num_realisation, 0, x)
        J_est = LQR_Ricatti.evaluate_J_X(out_t, out_x, dt)
        J_est_mean = torch.mean(J_est)
        V_init = LQR_Ricatti.v(out_t, out_x[0].transpose(-1, -2))[0,0]
        MSE_t.append((V_init-J_est_mean).numpy()**2)
    MSE_step.append(np.mean(MSE_t))
#print(np.mean(MSE))


# In[236]:


num_realisations = [10,50,100,500,1000,5000,10000,50000,100000]
timestep = 5000
MSE_path = []
tt = np.linspace(0,T,timestep+1)
St = LQR_Ricatti.solve_ricatti(tt).numpy()
dt = T/timestep
for N in num_realisations:
    MSE_N = []
    i = 0
    for x in ini_x[:]:
        i+=1
        print(i)
        out_t, out_x = LQR_Ricatti.simulate_X(timestep, N, 0, x)
        J_est = LQR_Ricatti.evaluate_J_X(out_t, out_x, dt)
        J_est_mean = torch.mean(J_est)
        V_init = LQR_Ricatti.v(out_t, out_x[0].transpose(-1, -2))[0,0]
        MSE_N.append((V_init-J_est_mean).numpy()**2)
    MSE_path.append(np.mean(MSE_N))


# In[234]:


plt.clf()
plt.figure(figsize=(10, 5))
ax1 = plt.subplot2grid(shape = (1,2), loc = (0,0))
ax1.plot(timesteps, MSE_step)
ax1.set_title('trainloss of v(t,x)')
plt.xlabel('number of simulation paths')
plt.ylabel('Mean-Square Error')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax2 = plt.subplot2grid(shape = (1,2), loc = (0,1))
ax2.plot(num_realisations, MSE_path)
ax2.set_title('trainloss of a(t,x)')
plt.xlabel('number of timesteps')
plt.ylabel('Mean-Square Error')
ax2.set_xscale('log')
ax2.set_yscale('log')
plt.savefig('MSEQ1.png')
ax2.semilogy()

