
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
from .utils import Reservoir, CBSR

from .common import MLP, ResNet18
import random
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

        self.ce = nn.CrossEntropyLoss()
        self.n_outputs = n_outputs

        self.opt = optim.SGD(self.parameters(), args.lr)

        self.n_memories = args.n_memories
        self.gpu = args.cuda

        # allocate episodic memory
        self.memory = CBSR(n_tasks * args.n_memories, (3, n_inputs, n_inputs), n_outputs)

        # allocate temporary synaptic memory
        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())
        self.grads = torch.Tensor(sum(self.grad_dims), n_tasks)
        if args.cuda:
            self.grads = self.grads.cuda()

        # allocate counters
        self.observed_tasks = []
        self.old_task = -1
        self.mem_cnt = 0

        self.nc_per_task = int(n_outputs / n_tasks)
        self.reference_gradients = None
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

        self.memory.update(x, y, t)
        # compute gradient on previous tasks
        if len(self.observed_tasks) > 1:
            self.zero_grad()
            m_x, m_y = self.memory.sample(sample_size=x.shape[0])
            output = self.net(m_x)
            logit_mask = self.label_mask[m_y]
            output = output * logit_mask
            ptloss = self.ce(output, m_y)
            ptloss.backward()
            self.reference_gradients = [
                p.grad.view(-1) if p.grad is not None
                else torch.zeros(p.numel(), device=self.net.device)
                for n, p in self.net.named_parameters()]
            self.reference_gradients = torch.cat(self.reference_gradients)
            self.opt.zero_grad()

        # now compute the grad on the current minibatch
        self.zero_grad()

        offset1, offset2 = compute_offsets(t, self.nc_per_task)
        loss = self.ce(self.forward(x, t)[:, offset1: offset2], y - offset1)
        loss.backward()

        # check if gradient violates constraints
        if len(self.observed_tasks) > 1:
            current_gradients = [
                p.grad.view(-1) if p.grad is not None
                else torch.zeros(p.numel(), device=self.net.device)
                for n, p in self.net.named_parameters()]
            current_gradients = torch.cat(current_gradients)

            assert current_gradients.shape == self.reference_gradients.shape, \
                "Different model parameters in AGEM projection"

            dotg = torch.dot(current_gradients, self.reference_gradients)
            if dotg < 0:
                alpha2 = dotg / torch.dot(self.reference_gradients,
                                          self.reference_gradients)
                grad_proj = current_gradients - \
                            self.reference_gradients * alpha2

                count = 0
                for n, p in self.net.named_parameters():
                    n_param = p.numel()
                    if p.grad is not None:
                        p.grad.copy_(grad_proj[count:count + n_param].view_as(p))
                    count += n_param

        self.opt.step()