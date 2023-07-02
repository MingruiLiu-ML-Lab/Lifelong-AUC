import numpy as np
import subprocess
import pickle
import torch
import os

cifar_path = "cifar-100-python.tar.gz"
mnist_path = "mnist.npz"
isic_data_path = "ISIC_2019_Training_Input.zip"
isic_label_path = "ISIC_2019_Training_GroundTruth.csv"
EuroSat_path = "EuroSat_path.zip"

# URL from: https://www.cs.toronto.edu/~kriz/cifar.html
if not os.path.exists(cifar_path):
    subprocess.call("wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz", shell=True)
    subprocess.call("tar xzfv cifar-100-python.tar.gz", shell=True)
#
# # URL from: https://github.com/fchollet/keras/blob/master/keras/datasets/mnist.py
if not os.path.exists(mnist_path):
    subprocess.call("wget https://s3.amazonaws.com/img-datasets/mnist.npz", shell=True)

if  not os.path.exists(isic_data_path):
    subprocess.call("wget https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip", shell=True)
    subprocess.call("unzip ISIC_2019_Training_Input.zip", shell=True)

if  not os.path.exists(isic_label_path):
    subprocess.call("wget https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Metadata.csv", shell=True)

if  not os.path.exists(EuroSat_path):
    subprocess.call("wget https://madm.dfki.de/files/sentinel/EuroSAT.zip", shell=True)
    subprocess.call("unzip EuroSat_path.zip", shell=True)



def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

cifar100_train = unpickle('cifar-100-python/train')
cifar100_test = unpickle('cifar-100-python/test')

x_tr = torch.from_numpy(cifar100_train[b'data'])
y_tr = torch.LongTensor(cifar100_train[b'fine_labels'])
x_te = torch.from_numpy(cifar100_test[b'data'])
y_te = torch.LongTensor(cifar100_test[b'fine_labels'])

torch.save((x_tr, y_tr, x_te, y_te), 'cifar100.pt')

f = np.load('mnist.npz')
x_tr = torch.from_numpy(f['x_train'])
y_tr = torch.from_numpy(f['y_train']).long()
x_te = torch.from_numpy(f['x_test'])
y_te = torch.from_numpy(f['y_test']).long()
f.close()

torch.save((x_tr, y_tr), 'mnist_train.pt')
torch.save((x_te, y_te), 'mnist_test.pt')
