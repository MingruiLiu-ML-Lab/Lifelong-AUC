
import argparse
import os.path
import torch

parser = argparse.ArgumentParser()

parser.add_argument('--i', default='raw/ISIC_data.pt', help='input directory')
parser.add_argument('--o', default='isic.pt', help='output file')
parser.add_argument('--n_tasks', default=4, type=int, help='number of tasks')
parser.add_argument('--seed', default=0, type=int, help='random seed')
args = parser.parse_args()

torch.manual_seed(args.seed)

tasks_tr = []
tasks_te = []

x_tr, y_tr, x_te, y_te = torch.load(os.path.join(args.i))

x_tr = (x_tr.float().view(x_tr.size(0), -1) - x_tr.min())/(x_tr.max() - x_tr.min())
x_te = (x_te.float().view(x_te.size(0), -1) - x_te.min())/(x_te.max() - x_te.min())
y_tr_new = torch.ones(y_tr.size(0))*(y_tr.max()+100)
y_te_new = torch.ones(y_te.size(0))*(y_tr.max()+100)
y_tr_new[torch.where(y_tr==1)]  = 0
y_tr_new[torch.where(y_tr==3)]  = 1
y_tr_new[torch.where(y_tr==0)]  = 2
y_tr_new[torch.where(y_tr==7)]  = 3
y_tr_new[torch.where(y_tr==2)]  = 4
y_tr_new[torch.where(y_tr==6)]  = 5
y_tr_new[torch.where(y_tr==4)]  = 6
y_tr_new[torch.where(y_tr==5)]  = 7

y_te_new[torch.where(y_te==1)]  = 0
y_te_new[torch.where(y_te==3)]  = 1
y_te_new[torch.where(y_te==0)]  = 2
y_te_new[torch.where(y_te==7)]  = 3
y_te_new[torch.where(y_te==2)]  = 4
y_te_new[torch.where(y_te==6)]  = 5
y_te_new[torch.where(y_te==4)]  = 6
y_te_new[torch.where(y_te==5)]  = 7



cpt = int(8 / args.n_tasks)

for t in range(args.n_tasks):
    c1 = t * cpt
    c2 = (t + 1) * cpt
    i_tr = ((y_tr_new >= c1) & (y_tr_new < c2)).nonzero().view(-1)
    i_te = ((y_te_new >= c1) & (y_te_new < c2)).nonzero().view(-1)
    if i_te.size(0)> 1000:
        i_te = i_te[:1000]
    tasks_tr.append([(c1, c2), x_tr[i_tr].clone(), y_tr_new[i_tr].clone()])
    tasks_te.append([(c1, c2), x_te[i_te].clone(), y_te_new[i_te].clone()])

torch.save([tasks_tr, tasks_te], args.o)
