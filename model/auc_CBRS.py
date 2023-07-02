
import torch
import torch.nn as nn
import torch.optim as optim
import random
import sys
sys.path.append('.')

from losses import AUCMLoss, AUCM_MultiLabel
from optimizer import PESG

import numpy as np
from copy import deepcopy
import torch.nn.functional as F
import torchvision
from .common import MLP, ResNet18
from sklearn.metrics import roc_auc_score
from .utils import Reservoir, RingBuffer, ClassBalanced, CBSR


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
        # 1. define W and V
        self.net = net
        self.net_v = deepcopy(net)

        self.n_outputs = n_outputs
        # self.imratio =  [0.068, 0.1329, 0.0768, 0.0957]

        # AUC LOSS + optimize (a, b, alpha)
        self.loss_fn_w = AUCMLoss(imratio=args.imratio)
        #self.loss_fn_v = AUCMLoss(imratio=0.5)
        self.loss_fn_v = nn.CrossEntropyLoss()

        #from libauc.optimizers import PESG
        self.opt_w = PESG(self.net, self.loss_fn_w.a, self.loss_fn_w.b, self.loss_fn_w.alpha, lr=args.lr, imratio = args.imratio)
        #self.opt_v = PESG(self.net_v, self.loss_fn_v.a, self.loss_fn_v.b, self.loss_fn_v.alpha, lr=args.lr, imratio = args.imratio)
        self.opt_v = optim.SGD(self.net_v.parameters(), args.lr)
        self.n_memories = args.n_memories
        self.gpu = args.cuda

        # allocate episodic memory
        self.memory = CBSR(n_tasks * self.n_memories, (3, n_inputs, n_inputs), n_outputs * 2)
        #self.memory = Reservoir(n_tasks * self.n_memories, (3, n_inputs, n_inputs))
        #self.memory = RingBuffer(n_tasks, self.n_memories, (3, n_inputs, n_inputs))

        # allocate counters
        self.observed_tasks = []
        self.old_task = -1

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

    def forward_v(self, x, t):
        output = self.net_v(x)
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
        flag = False
        if t != self.old_task:
            self.observed_tasks.append(t)
            self.old_task = t
            flag= True

        self.memory.update(x, y, t)

        # compute gradient on previous tasks
        loss_v = 0
        loss_wv = 0
        self.opt_w.zero_grad()
        self.opt_v.zero_grad()

        # sample memory
        m_x, m_y = self.memory.sample(sample_size=x.shape[0])
        output = self.net_v(m_x)
        #logit_mask = self.label_mask[m_y // 2]
        #score = torch.masked_select(output, logit_mask.bool())
        #loss_v = self.loss_fn_v(score, m_y % 2)
        logit_mask = self.label_mask[m_y]
        output = output * logit_mask
        loss_v = self.loss_fn_v(output, m_y)

        output = self.net(m_x)  # feature [B, 100]
        output_v = self.net_v(m_x)  # feature_v [B, 100]
        loss_wv = F.mse_loss(output_v, output)

        # current mini-batch to update W
        logits = self.forward(x, t)[:, t]
        loss_w = self.loss_fn_w(logits, y % 2)

        if loss_v != 0 and loss_w != 0:
            alpha = loss_v.item() / loss_w.item()
        else:
            alpha = 0

        # loss_var = F.mse_loss(self.loss_fn_w.a, self.loss_fn_v.a) + F.mse_loss(self.loss_fn_w.b, self.loss_fn_v.b) \
        #             + F.mse_loss(self.loss_fn_w.alpha, self.loss_fn_v.alpha)
        loss = loss_w + alpha * loss_v + 0.1*loss_wv# + 0.1 * loss_var
        loss.backward()

        #self.loss_fn_w.alpha.data -= 0.01 * self.loss_fn_w.alpha.grad.data
        #self.loss_fn_v.alpha.data -= 0.01 * self.loss_fn_v.alpha.grad.data
        # self.lmda.data += 0.001 * self.lmda.grad.data
        # self.lmda.grad = None

        self.opt_w.step()
        self.opt_v.step()
        # if flag:
        #     print('loss=', loss)