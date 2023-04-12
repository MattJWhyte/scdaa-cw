
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import torch


# SET UP BASE LQR ----------------------------------------------
from base_solver import *
# Makes all matrices and lqr_solver available


num_realisations = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]
timestep = 5000
MSE_path = []
batch_size = 10

tt = np.linspace(0, T, timestep + 1)
solver.solve_ricatti(tt)
dt = T / timestep
x_ini = torch.rand(batch_size, 1, 2) * 6 - 3.0
x_ini = x_ini.transpose(-1, -2)
t_ini = torch.rand(batch_size)*T

for N in num_realisations:
    MSE_N = []
    i = 0
    for t,x in zip(t_ini,x_ini[:]):
        print("N: {}, i: {}".format(N,i))
        i += 1
        if N <= 50000:
            out_t, out_x = solver.simulate_X(timestep, N, t, x)
            J_est = solver.evaluate_J_X(out_t, out_x, dt)
            J_est_mean = torch.mean(J_est)
        else:
            J_est_mean = 0.0
            for _ in range(2):
                out_t, out_x = solver.simulate_X(timestep, N//2, t, x)
                J_est = solver.evaluate_J_X(out_t, out_x, dt)
                J_est_mean += 0.5*torch.mean(J_est)
        V_init = solver.v(out_t, out_x[0].transpose(-1, -2))[0, 0]
        MSE_N.append((V_init - J_est_mean).numpy() ** 2)

    MSE_path.append(np.sqrt(np.mean(MSE_N)))


plt.scatter(np.log10(num_realisations), np.log10(MSE_path))
slope, intercept = np.polyfit(np.log10(num_realisations), np.log10(MSE_path), 1)
print("Slope", slope)
trendline_x = np.array([np.log10(num_realisations).min(), np.log10(num_realisations).max()])
trendline_y = slope * trendline_x + intercept
plt.plot(trendline_x, trendline_y, color='red')
plt.title('MC error')
plt.xlabel('Number of simulation paths')
plt.ylabel('RMSE')
plt.savefig('q1_2a_num_sim_paths.png')