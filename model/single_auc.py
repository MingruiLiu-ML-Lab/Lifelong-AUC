import sys
sys.path.append('.')
from losses import AUCMLoss
from optimizer import PESG

import torch
from .common import MLP, ResNet18


class Net(torch.nn.Module):

    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()
        nl, nh = args.n_layers, args.n_hiddens

        # setup network
        self.is_cifar_isic = (args.data_file == 'cifar100.pt' or args.data_file == 'isic.pt')
        if self.is_cifar_isic:
            self.net = ResNet18(n_outputs, args)
        else:
            self.net = MLP([n_inputs] + [nh] * nl + [n_outputs])

        # setup losses
        self.loss_fn = AUCMLoss(margin=0.1)

        # setup optimizer
        self.opt = PESG(self.net, self.loss_fn.a, self.loss_fn.b, self.loss_fn.alpha, lr=args.lr)

        if self.is_cifar_isic:
            self.nc_per_task = n_outputs / n_tasks
        else:
            self.nc_per_task = n_outputs
        self.n_outputs = n_outputs

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
        self.opt.zero_grad()
        if self.is_cifar_isic:
            offset1, offset2 = self.compute_offsets(t)
            self.loss_fn((self.net(x)[:, offset1: offset2]),
                     y - offset1).backward()
        else:
            self.loss_fn(self(x, t), y).backward()
        self.opt.step()
