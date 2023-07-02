
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from .common import MLP, ResNet18

import sys
import random
sys.path.append('.')
from losses import AUCMLoss
from optimizer import PESG
import torchvision

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
        nl, nh = args.n_layers, args.n_hiddens
        self.margin = args.memory_strength
        self.net = net

        self.loss_fn = AUCMLoss(imratio=args.imratio)
        #from libauc.optimizers import PESG
        self.opt = PESG(self.net, a=self.loss_fn.a, b=self.loss_fn.b, alpha=self.loss_fn.alpha,
                        lr=args.lr)
        # self.loss_fn = nn.CrossEntropyLoss()
        #
        # self.opt = optim.SGD(self.parameters(), args.lr)
        self.n_outputs = n_outputs
        self.n_memories = args.n_memories
        self.gpu = args.cuda
        # self.imratio =  [0.068, 0.1329, 0.0768, 0.0957]
        # allocate episodic memory
        self.memory_data = torch.zeros(
            n_tasks, self.n_memories, 3, n_inputs, n_inputs, dtype=torch.float)
        self.memory_labs = torch.zeros(n_tasks, self.n_memories, dtype=torch.long)
        if args.cuda:
            self.memory_data = self.memory_data.cuda()
            self.memory_labs = self.memory_labs.cuda()

        # allocate counters
        self.observed_tasks = []
        self.old_task = -1
        self.mem_cnt = 0

        self.nc_per_task = int(n_outputs / n_tasks)

        self.label_mask = torch.from_numpy(np.kron(np.eye(n_tasks, dtype=int),
                                 np.ones((self.nc_per_task, self.nc_per_task)))).cuda()

    def forward(self, x, t):
        output = self.net(x)
        # make sure we predict classes within the current task
        offset1 = int(t * self.nc_per_task)
        offset2 = int((t + 1) * self.nc_per_task)
        if offset1 > 0:
            output[:, :offset1].data.fill_(-10e10)
        if offset2 < self.n_outputs:
            output[:, offset2:self.n_outputs].data.fill_(-10e10)
        return output

    def observe(self, x, t, y):
        # update memory
        if t != self.old_task:
            self.observed_tasks.append(t)
            self.old_task = t

        # Update ring buffer storing examples from current task
        bsz = y.data.size(0)
        endcnt = min(self.mem_cnt + bsz, self.n_memories)
        effbsz = endcnt - self.mem_cnt
        self.memory_data[t, self.mem_cnt: endcnt].copy_(
            x.data[: effbsz])
        if bsz == 1:
            self.memory_labs[t, self.mem_cnt] = y.data[0]
        else:
            self.memory_labs[t, self.mem_cnt: endcnt].copy_(
                y.data[: effbsz])
        self.mem_cnt += effbsz
        if self.mem_cnt == self.n_memories:
            self.mem_cnt = 0

        # compute gradient on previous tasks
        loss_v = 0
        self.opt.zero_grad()
        # memory
        if len(self.observed_tasks) > 1:
            # Average sample
            sampler_per_task = self.n_memories // (len(self.observed_tasks))
            m_x, m_y = [], []
            for tt in range(len(self.observed_tasks) - 1):
                # fwd/bwd on the examples in the memory
                past_task = self.observed_tasks[tt]
                index = torch.randperm(len(self.memory_data[past_task]))[:sampler_per_task]
                m_x.append(self.memory_data[past_task][index])
                m_y.append(self.memory_labs[past_task][index])
            m_x, m_y = torch.cat(m_x), torch.cat(m_y)
            # shuffle
            index = torch.randperm(len(m_x))
            m_x, m_y = m_x[index], m_y[index]
            output = self.net(m_x)
            logit_mask = self.label_mask[m_y // 2]
            score = torch.masked_select(output, logit_mask.bool())
            loss_v = self.loss_fn(score, m_y % 2)

        loss = self.loss_fn(self.forward(x, t)[:, t], y % 2)

        if loss_v != 0 and loss != 0:
            alpha = loss_v.item() / loss.item()
        else:
            alpha = 0

        loss += alpha * loss_v

        loss.backward()

        self.opt.step()