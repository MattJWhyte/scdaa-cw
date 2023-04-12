
import numpy as np
import torch

def positive_def_matrix():
    A = np.random.randn(2,2)
    B = A + A.T
    M = (B.T @ B) / 4.0 + 0.00001 * np.eye(2)
    return M


def torchify_this(bitch,use_cuda=True):
    t = torch.from_numpy(bitch).float()
    return cuda(t) if use_cuda else t


USE_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if USE_CUDA else "cpu"

def cuda(x):
    return x.to(DEVICE)


def train_net(net, loss_func, batch_size, num_epochs, lr, threshold=None):
    print("Train commencing ...")
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=(1000,2000,3000,4000,5000,), gamma=0.1)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

    for it in range(num_epochs):
        optimizer.zero_grad()
        loss_a = loss_func.loss_func(batch_size)
        if isinstance(loss_a, tuple) or isinstance(loss_a, list):
            loss = loss_a[0]
        else:
            loss = loss_a

        if it % (int(num_epochs * 0.1)) == 0:
            print("{}% complete ... loss: {}".format(round(it/float(num_epochs)*100), loss))
            print(loss_a)

        if threshold is not None and loss <= threshold:
            print("{}% complete (early stop) ... loss: {}".format(round(it / float(num_epochs) * 100), loss))
            return True

        loss.backward()
        optimizer.step()
        #scheduler.step()

    return False