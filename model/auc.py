
import torch
import torch.nn as nn
import torch.optim as optim
import random
import sys
sys.path.append('.')

from losses import AUCMLoss
from optimizer import PESG

import numpy as np
from copy import deepcopy
import torch.nn.functional as F
import torchvision
from .common import MLP, ResNet18
from sklearn.metrics import roc_auc_score
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
        # 1. define W and V
        self.net = net
        self.net_v = deepcopy(net)

        self.n_outputs = n_outputs
        # self.imratio =  [0.068, 0.1329, 0.0768, 0.0957]

        # AUC LOSS + optimize (a, b, alpha)
        self.loss_fn_w = AUCMLoss(imratio=args.imratio)
        self.loss_fn_v = AUCMLoss(imratio=args.imratio)
        #from libauc.optimizers import PESG
        self.opt_w = PESG(self.net, self.loss_fn_w.a, self.loss_fn_w.b, self.loss_fn_w.alpha, lr=args.lr, imratio = args.imratio)
        self.opt_v = PESG(self.net_v, self.loss_fn_v.a, self.loss_fn_v.b, self.loss_fn_v.alpha, lr=args.lr, imratio = args.imratio)
        self.n_memories = args.n_memories
        self.gpu = args.cuda

        # allocate episodic memory
        self.memory_data = torch.zeros(
            n_tasks, self.n_memories, 3, n_inputs, n_inputs, dtype=torch.float)
        self.memory_labs = torch.zeros(n_tasks, self.n_memories, dtype=torch.long)
        if args.cuda:
            self.memory_data = self.memory_data.cuda()
            self.memory_labs = self.memory_labs.cuda()

        self.lmda = torch.tensor(1, dtype=torch.float32, device=self.memory_data.device, requires_grad=False)
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

    def distill_loss(self, y_s, y_t, t=4):
        p_s = F.log_softmax(y_s / t, dim=1)
        p_t = F.softmax(y_t / t, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='sum') * (t ** 2) / y_s.shape[0]
        return loss

    def observe(self, x, t, y):
        # update memory
        flag = False
        if t != self.old_task:
            self.observed_tasks.append(t)
            self.old_task = t
            flag= True

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
        loss_wv = 0
        self.opt_w.zero_grad()
        self.opt_v.zero_grad()

        # memory
        if len(self.observed_tasks) > 1:
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
            output = self.net_v(m_x)
            logit_mask = self.label_mask[m_y // 2]
            score = torch.masked_select(output, logit_mask.bool())
            loss_v = self.loss_fn_v(score, m_y % 2)

            output = self.net(m_x)  # feature [B, 100]
            output_v = self.net_v(m_x)# feature_v [B, 100]
            loss_wv = F.mse_loss(output_v, output)

        # current mini-batch to update W
        logits = self.forward(x, t)[:, t]
        loss_w = self.loss_fn_w(logits, y % 2)

        if loss_v != 0 and loss_w != 0:
            alpha = loss_v.item() / loss_w.item()
        else:
            alpha = 0
        beta = 0.1
        loss_var = F.mse_loss(self.loss_fn_w.a, self.loss_fn_v.a) + F.mse_loss(self.loss_fn_w.b, self.loss_fn_v.b) \
                    + F.mse_loss(self.loss_fn_w.alpha, self.loss_fn_v.alpha)
        loss = loss_w + alpha * loss_v + beta * loss_wv + beta * loss_var
        loss.backward()

        self.loss_fn_w.alpha.data -= 0.1 * beta  * self.loss_fn_w.alpha.grad.data
        self.loss_fn_v.alpha.data -= 0.1 * beta * self.loss_fn_v.alpha.grad.data
        # self.lmda.data += 0.001 * self.lmda.grad.data
        # self.lmda.grad = None


        self.opt_w.step()
        self.opt_v.step()
        # if flag:
        #     print('loss=', loss)