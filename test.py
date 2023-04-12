import scipy.stats
from scipy.integrate import odeint
import numpy as np
from lqr_solver import LQR_Solver
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

# SET UP LQR ----------------------------------------------

H = np.eye(2) #positive_def_matrix()
M = np.eye(2) #positive_def_matrix() #np.array([[0.15663973 0.15513884],[0.15513884 0.20362521]])

sigma = 0.05*np.eye(2) #np.array([[0.05, 0.0],[0.05,0.0]])#0.05*np.eye(2) #positive_def_matrix()
T = 1
C = 0.1*np.eye(2) #positive_def_matrix()
D = 0.1*np.eye(2) #positive_def_matrix()
R = np.eye(2) #positive_def_matrix()

tt = np.linspace(0,T,5000+1)

N = 5000

LQR_Ricatti = LQR_Solver(H, M, sigma, T, C, D, R)
'''
LQR_Ricatti.solve_ricatti(tt)



Out_tt, Out_t_init, Out_x_init, Out_x, mask = LQR_Ricatti.simulate_X_multi_init(N, n_realisations=10**4)

J_est = LQR_Ricatti.evaluate_J_X(Out_tt, Out_x, Out_tt[1]-Out_tt[0], mask=mask)
v_true = LQR_Ricatti.v(Out_t_init, Out_x_init.transpose(-1,-2)).squeeze(-1)

J_est_mean = torch.mean(J_est)
print("Error:", (J_est_mean-v_true)**2)
diff = torch.abs(J_est-v_true)
print(torch.mean(diff, dim=0))
print(torch.max(diff))

plt.plot(Out_x[0,:,0,0], Out_x[0,:,1,0])
plt.scatter(Out_x[0,0,0,0], Out_x[0,0,1,0], label="start")
plt.scatter(Out_x[0,-1,0,0], Out_x[0,-1,1,0], label="end")
plt.legend()
plt.savefig("traj.png")
'''

'''
N = 5000
tt = np.linspace(0,T,N+1)

avg = [0]
M = 50000


LQR_Ricatti.solve_ricatti(tt)
v = None

for i in range(1000):
    print("iter {}".format(i))
    num_realisations = M
    x_init = torch.ones((2, 1))

    Out_tt, Out_x = LQR_Ricatti.simulate_X(N, num_realisations, 0.0, x_init)

    dt = T / N

    J_est = LQR_Ricatti.evaluate_J_X(Out_tt, Out_x, dt)

    J_est_mean = torch.mean(J_est)
    avg.append((avg[-1]*i + J_est_mean)/(i+1))

    V_init = LQR_Ricatti.v(Out_tt, Out_x[0].transpose(-1, -2))[0, 0]
    if v is not None:
        print(v == V_init)
    v = V_init

    print("iter {}: avg {}, diff {}".format(i,avg[-1],avg[-1]-V_init))

    plt.clf()
    plt.plot([np.log(m) for m in range(1,i+2)], [np.log(np.abs(v-e)) for e in avg[1:]])
    plt.scatter([np.log(m) for m in range(1,i+2)], [np.log(np.abs(v-e)) for e in avg[1:]])
    plt.savefig("running-avg.png")
'''

Meta_M_mean = []

for K in range(100):
    N = 5000
    tt = np.linspace(0,T,N+1)

    M_list = [10,50,100,500,1000,5000,10000,50000]
    M_val = []
    M_mean = []

    LQR_Ricatti.solve_ricatti(tt)


    for M in M_list:

        #St = LQR_Ricatti.solve_ricatti(tt).numpy()

        num_realisations = M
        x_init = torch.ones((2,1))

        Out_tt, Out_x = LQR_Ricatti.simulate_X(N, num_realisations, 0.0, x_init)

        dt = T/N

        J_est = LQR_Ricatti.evaluate_J_X(Out_tt, Out_x, dt)

        J_est_mean = torch.mean(J_est)
        M_mean.append(M)
        V_init = LQR_Ricatti.v(Out_tt, Out_x[0].transpose(-1, -2))[0,0]

        print("V init", V_init)
        print("J est", J_est_mean)

        #D = torch.mean((J_est-V_init)**2)
        D = (J_est_mean-V_init)**2
        M_val.append(D.numpy())

    from scipy.stats import linregress

    log_M = [np.log(m) for m in M_list]
    log_D = [np.log(e) for e in M_val]


    lr = linregress(log_M, log_D)

    fit = lr.intercept + lr.slope*np.array(log_M)

    plt.plot([np.log(m) for m in M_list],[np.log(e) for e in M_val])
    plt.scatter([np.log(m) for m in M_list],[np.log(e) for e in M_val])
    plt.plot(log_M, fit)
    plt.savefig("MC-error.png")

    print("SLOPE: ", lr.slope)

    Meta_M_mean = []


