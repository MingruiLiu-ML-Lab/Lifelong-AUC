
from torchvision import transforms
from PIL import Image
import argparse
import os.path
import random
import torch

def rotate_dataset(d, rotation):
    rot = transforms.RandomRotation((rotation, rotation))
    result = torch.zeros((d.size(0),3,32,32))
    for i in range(d.size(0)):
        rot_result = rot.forward(d[i])
        result[i] = rot_result
    return result



parser = argparse.ArgumentParser()

parser.add_argument('--i', default='raw/ISIC_data.pt', help='input directory')
parser.add_argument('--o', default='isic_rotations.pt', help='output file')
parser.add_argument('--n_tasks', default=10, type=int, help='number of tasks')
parser.add_argument('--min_rot', default=0.,
                    type=float, help='minimum rotation')
parser.add_argument('--max_rot', default=360.,
                    type=float, help='maximum rotation')
parser.add_argument('--seed', default=0, type=int, help='random seed')
args = parser.parse_args()

torch.manual_seed(args.seed)

tasks_tr = []
tasks_te = []

x_tr, y_tr, x_te, y_te = torch.load(os.path.join(args.i))


for t in range(args.n_tasks):

    min_rot = 1.0 * t / args.n_tasks * (args.max_rot - args.min_rot) + \
        args.min_rot
    max_rot = 1.0 * (t + 1) / args.n_tasks * \
        (args.max_rot - args.min_rot) + args.min_rot
    rot = random.random() * (max_rot - min_rot) + min_rot
    x_tr_new = rotate_dataset(x_tr, rot)
    x_te_new = rotate_dataset(x_te, rot)
    x_tr_new = x_tr_new.float().view(x_tr_new.size(0), -1)
    x_te_new = x_te_new.float().view(x_te_new.size(0), -1)
    tasks_tr.append([(0, 8), x_tr_new, y_tr])
    tasks_te.append([(0, 8), x_te_new, y_te])

torch.save([tasks_tr, tasks_te], args.o)