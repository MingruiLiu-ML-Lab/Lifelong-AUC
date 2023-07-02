
import random

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import torchvision.models
from .utils import CBSR
from .common import MLP, ResNet18

# Auxiliary functions useful for GEM's inner optimization.

def compute_offsets(task, nc_per_task):
    """
        Compute offsets for cifar to determine which
        outputs to select for a given task.
    """

    offset1 = task * nc_per_task
    offset2 = (task + 1) * nc_per_task
    return offset1, offset2


class Net(nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 net,
                 args):
        super(Net, self).__init__()
        self.net = net
        self.ce = nn.CrossEntropyLoss()
        self.n_outputs = n_outputs

        self.opt = optim.SGD(self.parameters(), args.lr)

        self.n_memories = args.n_memories
        self.gpu = args.cuda

        self.memory = CBSR(n_tasks * self.n_memories, (3, n_inputs, n_inputs), n_outputs)

        # allocate counters
        self.observed_tasks = []
        self.old_task = -1
        self.mem_cnt = 0

        self.nc_per_task = int(n_outputs / n_tasks)
        self.label_mask = torch.from_numpy(np.kron(np.eye(n_tasks, dtype=int),
                                 np.ones((self.nc_per_task, self.nc_per_task)))).cuda()
        self.n_cls_seen = 0

    def forward(self, x, t):
        output = self.net(x)
        offset1 = int(t * self.nc_per_task)
        offset2 = int((t + 1) * self.nc_per_task)
        if offset1 > 0:
            output[:, :offset1].data.fill_(-10e10)
        if offset2 < self.n_outputs:
            output[:, offset2:self.n_outputs].data.fill_(-10e10)
        return output

    def observe(self, x, t, y):
        # update memory
        flag = False
        if t != self.old_task:
            self.observed_tasks.append(t)
            self.old_task = t
            flag=True
            self.n_cls_seen += self.nc_per_task
        self.memory.update(x, y, t)

        # compute gradient on previous tasks
        loss_v = 0
        self.zero_grad()
        # memory
        m_x, m_y = self.memory.sample(sample_size=x.shape[0])

        output = self.net(m_x)
        logit_mask = self.label_mask[m_y]
        output = output * logit_mask
        loss_v = self.ce(output, m_y)

        # now compute the grad on the current minibatch

        offset1, offset2 = compute_offsets(t, self.nc_per_task)
        loss_w = self.ce(self.forward(x, t)[:, offset1: offset2], y - offset1)

        alpha = 1 / self.n_cls_seen
        loss = alpha * loss_w + (1 - alpha) * loss_v

        loss.backward()

        self.opt.step()
