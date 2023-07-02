
import torch
import torch.nn as nn
from .common import MLP, ResNet18
import torchvision
from .utils import Reservoir

class Net(torch.nn.Module):

    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 net,
                 args):
        super(Net, self).__init__()
        nl, nh = args.n_layers, args.n_hiddens
        self.is_cifar_isic = True
        # setup network
        self.net = net

        # setup optimizer
        self.opt = torch.optim.SGD(self.parameters(), lr=args.lr)

        # setup losses
        self.bce = torch.nn.CrossEntropyLoss()

        if self.is_cifar_isic:
            self.nc_per_task = n_outputs / n_tasks
        else:
            self.nc_per_task = n_outputs
        self.n_outputs = n_outputs
        self.memory = Reservoir(n_tasks * args.n_memories, (3, n_inputs, n_inputs))

    def compute_offsets(self, task):
        if self.is_cifar_isic:
            offset1 = task * self.nc_per_task
            offset2 = (task + 1) * self.nc_per_task
        else:
            offset1 = 0
            offset2 = self.n_outputs
        return int(offset1), int(offset2)

    def forward(self, x, t):
        output = self.net(x)
        if self.is_cifar_isic:
            # make sure we predict classes within the current task
            offset1, offset2 = self.compute_offsets(t)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, offset2:self.n_outputs].data.fill_(-10e10)
        return output

    def observe(self, x, t, y):
        self.train()
        self.zero_grad()
        self.memory.update(x, y, t)
        m_x, m_y = self.memory.sample(sample_size=x.shape[0])
        loss = self.bce(self.net(m_x), m_y)
        loss.backward()
        self.opt.step()
