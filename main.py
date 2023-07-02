
import importlib
import datetime
import argparse
import random
import uuid
import time
import os
import torchvision

import numpy as np

import torch
from metrics.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from data.generate_imbalanced_dataset import ImbalanceGenerator
from model.common import ResNet18

# from libauc import optimizers
# continuum iterator #########################################################


def to_tensor(x, y, deivce):
    return torch.tensor(x, device=deivce), torch.tensor(y, device=deivce),


def load_datasets(args):
    d_tr, d_te = torch.load(args.data_path + '/' + args.data_file)
    n_inputs = d_tr[0][1].size(1)
    n_outputs = -1
    for i in range(len(d_tr)):
        if args.make_imbalanced == 'yes':
            # imratio = 0.05
            X_tr, Y_tr = ImbalanceGenerator(d_tr[i][1].cpu().numpy(), d_tr[i][2].cpu().numpy(), is_balanced=False,
                                            imratio=args.imratio)
            X_te, Y_te = ImbalanceGenerator(d_te[i][1].cpu().numpy(), d_te[i][2].cpu().numpy(), is_balanced=True,
                                            imratio=args.imratio)
            Y_tr += n_outputs + 1
            Y_te += n_outputs + 1
            d_tr[i][1], d_tr[i][2] = to_tensor(X_tr, Y_tr, d_tr[i][1].device)
            d_te[i][1], d_te[i][2] = to_tensor(X_te, Y_te, d_te[i][1].device)
            d_tr[i][0] = (min(Y_tr), max(Y_tr) + 1)
            d_te[i][0] = (min(Y_te), max(Y_te) + 1)

        n_outputs = int(max(n_outputs, d_tr[i][2].max().item()))
        n_outputs = int(max(n_outputs, d_te[i][2].max().item()))

    return d_tr, d_te, n_inputs, n_outputs + 1, len(d_tr)


class Continuum:

    def __init__(self, data, args):
        self.data = data
        self.batch_size = args.batch_size
        n_tasks = len(data)
        task_permutation = range(n_tasks)

        if args.shuffle_tasks == 'yes':
            task_permutation = torch.randperm(n_tasks).tolist()

        sample_permutations = []

        for t in range(n_tasks):
            N = data[t][1].size(0)
            if args.samples_per_task <= 0:
                n = N
            else:
                n = min(args.samples_per_task, N)

            p = torch.randperm(N)[0:n]
            sample_permutations.append(p)

        self.permutation = []

        for t in range(n_tasks):
            task_t = task_permutation[t]
            for _ in range(args.n_epochs):
                task_p = [[task_t, i] for i in sample_permutations[task_t]]
                random.shuffle(task_p)
                self.permutation += task_p

        self.length = len(self.permutation)
        self.current = 0

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        if self.current >= self.length:
            raise StopIteration
        else:
            ti = self.permutation[self.current][0]
            j = []
            i = 0
            while (((self.current + i) < self.length) and
                   (self.permutation[self.current + i][0] == ti) and
                   (i < self.batch_size)):
                j.append(self.permutation[self.current + i][1])
                i += 1
            self.current += i
            j = torch.LongTensor(j)
            return self.data[ti][1][j], ti, self.data[ti][2][j]


# train handle ###############################################################


def eval_tasks(model, tasks, args):
    model.eval()
    result = []
    auc_result = []
    for i, task in enumerate(tasks):
        t = i
        x = task[1]
        y = task[2]
        rt = 0

        eval_bs = x.size(0)
        x = x.view(x.shape[0], 3, 32, 32)
        y_true = []
        y_score = []
        for b_from in range(0, x.size(0), eval_bs):
            b_to = min(b_from + eval_bs, x.size(0) - 1)
            if b_from == b_to:
                xb = x[b_from]
                yb = torch.LongTensor([y[b_to]]).view(1, -1)
            else:
                xb = x[b_from:b_to]
                yb = y[b_from:b_to]

            with torch.no_grad():
                if args.cuda:
                    xb = xb.cuda()
                if 'auc' in args.model:
                    out = model.net(xb)[:, t]
                    score = torch.sigmoid(out)
                    pb = (score > 0.5).cpu()
                    rt += (pb == (yb % 2)).float().sum()
                    y_true.append(yb.numpy() % 2)
                    y_score.append(score.cpu().numpy())  # (0, 1) the last class is positive
                else:
                    out = model(xb, t)
                    pred = torch.softmax(out, dim=1)
                    # pred = torch.sigmoid(out)
                    _, pb = torch.max(out.data.cpu(), 1, keepdim=False)
                    rt += (pb == yb).float().sum()

                    y_true.append(yb.numpy() % 2)
                    y_score.append(pred.data.cpu().numpy()[:, (2 * i) + 1])  # (0, 1) the last class is positive
                # y_score.append(pred.data.cpu().numpy()[:, -1])

        y_true = np.concatenate(y_true, axis=0)
        y_score = np.concatenate(y_score, axis=0)
        result.append(rt / x.size(0))
        # calc auc one vs rest, rest classes are treated as negative
        auc_result.append(roc_auc_score(y_true, y_score))

    return result, auc_result


def life_experience(model, continuum, x_te, args):
    result_a = []
    result_auc = []
    result_t = []

    current_task = 0
    time_start = time.time()

    for (i, (x, t, y)) in enumerate(continuum):
        # commented: only eval the last task
        # if(((i % args.log_every) == 0) or (t != current_task)):
        #     res_acc, res_auc = eval_tasks(model, x_te, args)
        #     result_a.append(res_acc)
        #     result_auc.append(res_auc)
        #     result_t.append(current_task)
        #     current_task = t

        v_x = x.view(x.shape[0], 3, 32, 32)
        v_y = y.long()

        if args.cuda:
            v_x = v_x.cuda()
            v_y = v_y.cuda()

        model.train()
        model.observe(v_x, t, v_y)

    res_acc, res_auc = eval_tasks(model, x_te, args)
    result_a.append(res_acc)
    result_auc.append(res_auc)
    result_t.append(current_task)

    time_end = time.time()
    time_spent = time_end - time_start

    print('auc=', torch.Tensor(result_auc[-1]).mean().item())

    return torch.Tensor(result_t), torch.Tensor(result_a), torch.Tensor(result_auc), time_spent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Continuum learning')

    # model parameters
    parser.add_argument('--model', type=str, default='auc',
                        help='model to train')
    parser.add_argument('--n_hiddens', type=int, default=100,
                        help='number of hidden neurons at each layer')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='number of hidden layers')

    # memory parameters
    parser.add_argument('--n_memories', type=int, default=128,
                        help='number of memories per task')
    parser.add_argument('--memory_strength', default=0.5, type=float,
                        help='memory strength (meaning depends on memory)')
    parser.add_argument('--finetune', default='no', type=str,
                        help='whether to initialize nets in indep. nets')

    # optimizer parameters
    parser.add_argument('--n_epochs', type=int, default=1,
                        help='Number of epochs per task')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='SGD learning rate')

    # experiment parameters
    parser.add_argument('--cuda', type=str, default='yes',
                        help='Use GPU?')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    parser.add_argument('--imratio', type=float, default=0.05,
                        help='the proporation of positive samples')
    parser.add_argument('--log_every', type=int, default=100,
                        help='frequency of logs, in minibatches')
    parser.add_argument('--save_path', type=str, default='results/',
                        help='save models at the end of training')

    # data parameters
    parser.add_argument('--dataset', default='CIFAR100',
                        help='data file')
    parser.add_argument('--data_path', default='data/',
                        help='path where data is located')
    parser.add_argument('--data_file', default='cifar100.pt',
                        help='data file')
    parser.add_argument('--samples_per_task', type=int, default=-1,
                        help='training samples per task (all if negative)')
    parser.add_argument('--shuffle_tasks', type=str, default='no',
                        help='present tasks in order')
    parser.add_argument('--make_imbalanced', type=str, default='yes',
                        help='make data imbalanced')
    args = parser.parse_args()

    args.cuda = True if args.cuda == 'yes' else False
    args.finetune = True if args.finetune == 'yes' else False

    # multimodal model has one extra layer
    if args.model == 'multimodal':
        args.n_layers -= 1

    # unique identifier
    uid = uuid.uuid4().hex

    # initialize seeds
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)

    # load data
    x_tr, x_te, n_inputs, n_outputs, n_tasks = load_datasets(args)
    # set up continuum
    continuum = Continuum(x_tr, args)

    if 'auc' in args.model: # a score for each task
        n_outputs = n_outputs // 2

    # make network
    if args.dataset in ['AWA2', 'CUB200']:
        net = torchvision.models.resnet18(pretrained=True)
        net.fc = torch.nn.Linear(512, n_outputs)
    else:
        net = ResNet18(nclasses=n_outputs, args=args)
        n_inputs //= 3
        n_inputs = int(np.sqrt(n_inputs))

    Model = importlib.import_module('model.' + args.model)
    model = Model.Net(n_inputs, n_outputs, n_tasks, net, args)
    if args.cuda:
        model.cuda()

    # run model on continuum
    result_t, result_a, result_auc, spent_time = life_experience(
        model, continuum, x_te, args)

    # prepare saving path and file name
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    fname = args.model + '_' + args.data_file + '_'
    fname += datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    fname += '_' + uid
    fname = os.path.join(args.save_path, fname)

    # save confusion matrix and print one line of stats
    stats = confusion_matrix(result_t, result_a, result_auc, fname + '.txt')
    one_liner = str(vars(args)) + ' # '
    one_liner += ' '.join(["%.3f" % stat for stat in stats])
    print(fname + ': ' + one_liner + ' # ' + str(spent_time))

    # save all results in binary file
    torch.save((result_t, result_a, model.state_dict(),
                stats, one_liner, args), fname + '.pt')
