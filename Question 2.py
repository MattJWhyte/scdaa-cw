#!/usr/bin/env python
# coding: utf-8

# In[39]:


import matplotlib.pyplot as plt
from FFN import *
from DGM import *
import torch
import lqr_solver
from helper import positive_def_matrix

# NB : Ricatti must be solved first
def get_samples(solver, num_samples):
    t = torch.rand(num_samples) * solver.T
    x = torch.rand(num_samples, 1, 2) * 6 - 3.0
    return t, x, solver.v(t, x)

# SET UP LQR ----------------------------------------------

H = np.eye(2) #positive_def_matrix()
M = np.eye(2) #positive_def_matrix() #np.array([[0.15663973 0.15513884],[0.15513884 0.20362521]])

sigma = np.array([[0.05, 0.0],[0.05,0.0]])#0.05*np.eye(2) #positive_def_matrix()
T = 1
C = 0.1 * np.eye(2) #positive_def_matrix()
D = 0.1*np.eye(2) #positive_def_matrix()
R = np.eye(2) #positive_def_matrix()

lqr_s = lqr_solver.LQR_Solver(H, M, sigma, T, C, D, R)

# SOLVE RICATTI OVER GRID ------------------------------------

N = 10000
tt = np.linspace(0,T,N+1)
lqr_s.solve_ricatti(tt)
lqr_s.test_integrals()

print("Ricatti error", lqr_s.test_Ricatti_ODE())

# ML TIME -------------------------------------------------

net = Net_DGM(dim_x=2, dim_S=100)

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=(10000,), gamma=0.1)
loss_fn = nn.MSELoss()

batch_size = 2048
num_epochs = 10000
loss_track_v = []

for it in range(num_epochs):
    optimizer.zero_grad()

    train_t, train_x, train_v = get_samples(lqr_s, batch_size)

    train_t = train_t.unsqueeze(1)
    train_x = train_x.squeeze(1)

    pred_v = net(train_t, train_x)
    loss = loss_fn(pred_v, train_v)
    loss_track_v.append(loss.detach().numpy())

    if it % 10 == 0:
        print(torch.mean(train_v))
        print(torch.max(train_v))
        print("Epoch ", it, " loss ", loss)
    

    loss.backward()
    optimizer.step()
    scheduler.step()

plt.clf()

plt.plot([i for i in range(num_epochs)], loss_track_v)
plt.savefig("loss_2.1.png")


# In[41]:


#train a(t,x)--------------------------------------
# NB : Ricatti must be solved first
def get_a_samples(solver, num_samples):
    t = torch.rand(num_samples) * solver.T
    x = torch.rand(num_samples, 1, 2) * 6 - 3.0
    return t, x, solver.a(t, x)


# SOLVE RICATTI OVER GRID ------------------------------------

N = 10000
tt = np.linspace(0,T,N+1)
lqr_s.solve_ricatti(tt)
lqr_s.test_integrals()

print("Ricatti error", lqr_s.test_Ricatti_ODE())

# ML TIME -------------------------------------------------

net = FFN(sizes=[3, 100, 100, 2])

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=(10000,), gamma=0.1)
loss_fn = nn.MSELoss()

batch_size = 2048

loss_track_a = []

num_epochs = 10000

for it in range(num_epochs):
    optimizer.zero_grad()

    train_t, train_x, train_a = get_a_samples(lqr_s, batch_size)

    train_t = train_t.unsqueeze(1)
    train_x = train_x.squeeze(1)

    train_tx = torch.cat([train_t, train_x], 1)
    pred_a = net(train_tx)
    loss = loss_fn(pred_a, train_a)

    loss_track_a.append(loss.detach().numpy())

    if it % 10 == 0:
        print(torch.mean(torch.abs(train_a)))
        print(torch.mean(torch.abs(pred_a)))
        print("Epoch ", it, " loss ", loss)

    loss.backward()
    optimizer.step()
    scheduler.step()

plt.clf()

plt.plot([i for i in range(num_epochs)], loss_track_a)
plt.savefig("loss_2.2.png")


# In[42]:



plt.clf()
plt.figure(figsize=(10, 5))
ax1 = plt.subplot2grid(shape = (1,2), loc = (0,0))
ax1.plot([i for i in range(num_epochs)], loss_track_v)
ax1.set_title('trainloss of v(t,x)')
plt.xlabel('number of epoch')
plt.ylabel('trainloss')
ax1.set_yscale('log')
ax2 = plt.subplot2grid(shape = (1,2), loc = (0,1))
ax2.plot([i for i in range(num_epochs)],loss_track_a)
ax2.set_title('trainloss of a(t,x)')
plt.xlabel('number of epoch')
plt.ylabel('trianloss')
ax2.set_yscale('log')
plt.savefig('loss_Q2(log).png')


# In[43]:


plt.clf()
plt.figure(figsize=(10, 5))
ax1 = plt.subplot2grid(shape = (1,2), loc = (0,0))
ax1.plot([i for i in range(num_epochs)], loss_track_v)
ax1.set_title('trainloss of v(t,x)')
plt.xlabel('number of epoch')
plt.ylabel('trainloss')
ax2 = plt.subplot2grid(shape = (1,2), loc = (0,1))
ax2.plot([i for i in range(num_epochs)], loss_track_a)
ax2.set_title('trainloss of a(t,x)')
plt.xlabel('number of epoch')
plt.ylabel('trianloss')
plt.savefig('loss_Q2.png')


# In[44]:


plt.clf()
plt.figure(figsize=(10, 5))
ax1 = plt.subplot2grid(shape = (1,2), loc = (0,0))
ax1.plot([i for i in range(num_epochs)], loss_track_v)
ax1.set_title('trainloss of v(t,x)')
plt.xlabel('number of epoch')
plt.ylabel('trainloss')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax2 = plt.subplot2grid(shape = (1,2), loc = (0,1))
ax2.plot([i for i in range(num_epochs)], loss_track_a)
ax2.set_title('trainloss of a(t,x)')
plt.xlabel('number of epoch')
plt.ylabel('trainloss')
ax2.set_xscale('log')
ax2.set_yscale('log')
plt.savefig('loss_Q2(log-log).png')


# In[ ]:




