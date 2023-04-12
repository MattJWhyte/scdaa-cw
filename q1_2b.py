
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import torch


# SET UP BASE LQR ----------------------------------------------
from base_solver import *
# Makes all matrices and lqr_solver available

# NUMBER OF SIMULATIONS ---------------------------------------

num_realisations = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]
timestep = 5000
MSE_path = []
batch_size = 100

tt = np.linspace(0, T, timestep + 1)
solver.solve_ricatti(tt)
dt = T / timestep

x_ini = torch.rand(batch_size, 1, 2) * 6 - 3.0
t_ini = torch.rand(batch_size)*T

solver.solve_ricatti(tt)
v_ini = solver.v(t_ini, x_ini)

x_ini = x_ini.transpose(-1, -2)

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
        MSE_N.append((v_ini[i-1] - J_est_mean).numpy() ** 2)

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

# NUMBER OF TIME STEPS ------------------------------------------------------

# generate x
batch_size = 100
x_ini = torch.rand(batch_size, 1, 2) * 6 - 3.0
t_ini = torch.rand(batch_size)*T

max_tt = np.linspace(0, T, 5001)
solver.solve_ricatti(max_tt)

v_ini = solver.v(t_ini, x_ini)
x_ini = x_ini.transpose(-1, -2)

# simulation for time steps
num_realisation = 100000
timesteps = [1, 10, 50, 100, 500, 1000, 5000]
MSE_step = []
for N in timesteps:
    tt = np.linspace(0, T, N + 1)
    # Approximate Ricatti over this time grid
    St = solver.solve_ricatti(tt).numpy()
    dt = T / N
    MSE_t = []
    i = 0
    for t,x in zip(t_ini,x_ini[:]):
        i += 1
        print("N: {}, i: {}".format(N, i))
        # Break up into two simulations and take average to avoid memory issues
        out_t, out_x = solver.simulate_X(N, num_realisation//2, t, x)
        J_est = solver.evaluate_J_X(out_t, out_x, dt)
        J_est_mean = 0.5*torch.mean(J_est)
        out_t, out_x = solver.simulate_X(N, num_realisation // 2, t, x)
        J_est = solver.evaluate_J_X(out_t, out_x, dt)
        J_est_mean += 0.5*torch.mean(J_est)
        #V_init = solver.v(out_t, out_x[0].transpose(-1, -2))[0, 0]
        MSE_t.append((v_ini[i-1] - J_est_mean).numpy() ** 2)

    MSE_step.append(np.sqrt(np.mean(MSE_t)))
# print(np.mean(MSE))

plt.clf()
plt.scatter(np.log10(timesteps), np.log10(MSE_step))
slope, intercept = np.polyfit(np.log10(timesteps), np.log10(MSE_step), 1)
print("Slope", slope)
trendline_x = np.array([np.log10(timesteps).min(), np.log10(timesteps).max()])
trendline_y = slope * trendline_x + intercept
plt.plot(trendline_x, trendline_y, color='red')
plt.title('MC error')
plt.xlabel('Number of timesteps')
plt.ylabel('RMSE')
plt.savefig('q1_2b_num_timesteps.png')



plt.clf()
plt.figure(figsize=(10, 5))
ax1 = plt.subplot2grid(shape=(1, 2), loc=(0, 0))
ax1.plot(np.log10(num_realisations), np.log10(MSE_path))
ax1.set_title('MC error')
plt.xlabel('Number of simulation paths')
plt.ylabel('Root Mean-Square Error')
ax2 = plt.subplot2grid(shape=(1, 2), loc=(0, 1))
ax2.plot(np.log10(timesteps), np.log10(MSE_step))
ax2.set_title('MC error')
plt.xlabel('Number of timesteps')
plt.ylabel('Root Mean-Square Error')
plt.savefig('MSEQ1.png')


np.save("timesteps.npy", np.array(timesteps))
np.save("mse_step.npy", np.array(MSE_step))
np.save("rum_realisations.npy", np.array(num_realisations))
np.save("mse_path.npy", np.array(MSE_path))