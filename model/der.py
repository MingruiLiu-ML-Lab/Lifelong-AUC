import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from .common import MLP, ResNet18
import random


from torch.nn import functional as F
from .utils_der import Buffer


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
        self.is_cifar_isic = (
                    args.data_file == 'cifar100.pt' or args.data_file == 'isic.pt' or args.data_file == 'isic_rotations.pt'\
                    or args.data_file == 'eurosat_rotations.pt' or args.data_file == 'eurosat_split.pt')

        self.net = net

        self.ce = nn.CrossEntropyLoss()
        self.n_outputs = n_outputs

        self.opt = optim.SGD(self.parameters(), args.lr)

        self.n_memories = args.n_memories
        self.gpu = args.cuda

        # allocate episodic memory
        self.memory_data = torch.FloatTensor(
            n_tasks, self.n_memories, n_inputs)
        self.memory_labs = torch.LongTensor(n_tasks, self.n_memories)
        if args.cuda:
            self.memory_data = self.memory_data.cuda()
            self.memory_labs = self.memory_labs.cuda()

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
        # if self.is_cifar_isic:
        #     self.nc_per_task = int(n_outputs / n_tasks)
        # else:
        #     self.nc_per_task = n_outputs
        self.nc_per_task = int(n_outputs / n_tasks)
        self.reference_gradients = None
        self.label_mask = torch.from_numpy(np.kron(np.eye(n_tasks, dtype=int),
                                                   np.ones((self.nc_per_task, self.nc_per_task)))).cuda()
    # def __init__(self, backbone, loss, args, transform):
    #     super(Der, self).__init__(backbone, loss, args, transform)
        self.device = 'cuda' if args.cuda == True else 'cpu'
        self.buffer = Buffer(args.n_memories, self.device)

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

        self.opt.zero_grad()

        # outputs = self.net(x)
        offset1, offset2 = compute_offsets(t, self.nc_per_task)
        outputs= self.forward(x, t)
        loss = self.ce(outputs[:, offset1: offset2], y - offset1)
        # loss = self.loss(outputs, y)

        if not self.buffer.is_empty():
            buf_inputs, buf_logits = self.buffer.get_data(
                self.n_memories, transform=None)
            buf_outputs = self.net(buf_inputs)
            loss += 0.5 * F.mse_loss(buf_outputs, buf_logits)

        loss.backward()
        # nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1, norm_type=2)
        self.opt.step()
        self.buffer.add_data(examples=x, logits=self.net(x).data)

