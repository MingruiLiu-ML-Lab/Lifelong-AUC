

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
from data.dataloader import VisionDataset
from model.common import ResNet18


def eval_tasks(model, task_loaders, args):
    model.eval()
    result = []
    auc_result = []
    for i, task_loader in enumerate(task_loaders):
        t = i
        y_true = []
        y_score = []
        rt = 0

        for xb, yb in task_loader:
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
                    _, pb = torch.max(out.data.cpu(), 1, keepdim=False)
                    rt += (pb == yb).float().sum()

                    y_true.append(yb.numpy() % 2)
                    y_score.append(pred.data.cpu().numpy()[:, (2 * i) + 1])  # (0, 1) the last class is positive
                # y_score.append(pred.data.cpu().numpy()[:, -1])

        y_true = np.concatenate(y_true, axis=0)
        y_score = np.concatenate(y_score, axis=0)
        result.append(rt / len(y_true))
        # calc auc one vs rest, rest classes are treated as negative
        try:
            auc_result.append(roc_auc_score(y_true, y_score))
        except:
            auc_result.append(0.5)

    return result, auc_result


def life_experience(model, dataset, args):
    result_a = []
    result_auc = []
    result_t = []

    current_task = 0
    time_start = time.time()

    for t, task_loader in enumerate(dataset.train_task_loaders):
        for x, y in task_loader:
            x = x.cuda()
            y = y.cuda()
            model.train()
            model.observe(x, t, y)
        #
        # res_acc, res_auc = eval_tasks(model, dataset.test_task_loaders, args)
        # result_a.append(res_acc)
        # result_auc.append(res_auc)
        # result_t.append(current_task)
        # current_task = t

    res_acc, res_auc = eval_tasks(model, dataset.test_task_loaders, args)
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
    parser.add_argument('--model', type=str, default='mega',
                        help='model to train')
    parser.add_argument('--n_hiddens', type=int, default=100,
                        help='number of hidden neurons at each layer')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='number of hidden layers')

    # memory parameters
    parser.add_argument('--n_memories', type=int, default=64,
                        help='number of memories per task')
    parser.add_argument('--memory_strength', default=0.5, type=float,
                        help='memory strength (meaning depends on memory)')
    parser.add_argument('--finetune', default='no', type=str,
                        help='whether to initialize nets in indep. nets')

    # optimizer parameters
    parser.add_argument('--n_epochs', type=int, default=1,
                        help='Number of epochs per task')
    parser.add_argument('--mem_batch_size', type=int, default=64,
                        help='number of memories per task')
    parser.add_argument('--batch_size', type=int, default=64,
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
    parser.add_argument('--data_path', default='data/AWA2',
                        help='path where data is located')
    parser.add_argument('--dataset', default='AWA2',
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
    args.data_file  = ''
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
    dataset = VisionDataset(args)
    n_outputs = dataset.num_classes
    n_tasks = dataset.num_tasks

    if 'auc' in args.model: # a score for each task
        n_outputs = n_outputs // 2

    # make network
    if args.dataset in ['AWA2', 'CUB200']:
        net = torchvision.models.resnet18(pretrained=True)
        net.fc = torch.nn.Linear(512, n_outputs)
        n_inputs = 224
    else:
        net = ResNet18(nclasses=n_outputs, args=args)
        n_inputs = 32

    # load model
    Model = importlib.import_module('model.' + args.model)
    model = Model.Net(n_inputs, n_outputs, n_tasks, net, args)
    if args.cuda:
        model.cuda()

    # run model on continuum
    result_t, result_a, result_auc, spent_time = life_experience(
        model, dataset, args)

    # prepare saving path and file name
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    fname = args.model + '_' + args.dataset + '_'
    fname += datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    fname += '_' + uid
    fname = os.path.join(args.save_path, fname)

    # save confusion matrix and print one line of stats
    stats = confusion_matrix(result_t, result_a, result_auc, fname + '.txt')
    one_liner = str(vars(args)) + ' # '
    one_liner += ' '.join(["%.3f" % stat for stat in stats])
    print(fname + ': ' + one_liner + ' # ' + str(spent_time))

    # save all results in binary file
    torch.save((result_t, result_a, result_auc, model.state_dict(),
                stats, one_liner, args), fname + '.pt')
