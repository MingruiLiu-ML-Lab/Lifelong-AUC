# from libauc.datasets.data_generator import ImbalanceGenerator


'''
Currently only support for binary classification
'''
import numpy as np
import argparse
import os.path
import torch


def check_imbalance_binary(Y):
    # numpy array
    num_samples = len(Y)
    pos_count = np.count_nonzero(Y == 1)
    neg_count = np.count_nonzero(Y == 0)
    pos_ratio = pos_count / (pos_count + neg_count)
    print('#SAMPLES: [%d], POS:NEG: [%d : %d], POS RATIO: %.4f' % (num_samples, pos_count, neg_count, pos_ratio))


def ImbalanceGenerator(X, Y, imratio=0.1, shuffle=False, is_balanced=False, random_seed=123):
    '''
    Imbalanced Data Generator
    '''

    assert isinstance(X, (np.ndarray, np.generic)), 'data needs to be numpy type!'
    assert isinstance(Y, (np.ndarray, np.generic)), 'data needs to be numpy type!'

    #num_classes = np.unique(Y).size
    min_y = np.min(Y)
    max_y = np.max(Y)
    num_classes = max_y - min_y + 1
    if num_classes == 2:
        split_index = 0 + min_y
    else:
        split_index = (num_classes // 2) + min_y

    # shuffle before preprocessing (add randomness for removed samples)
    id_list = list(range(X.shape[0]))
    np.random.seed(random_seed)
    np.random.shuffle(id_list)
    X = X[id_list]
    Y = Y[id_list]
    X_copy = X.copy()
    Y_copy = Y.copy()
    Y_copy[Y_copy <= split_index] = 0  # [0, ....]
    Y_copy[Y_copy >= split_index + 1] = 1  # [0, ....]

    if is_balanced == False:
        num_neg = np.where(Y_copy == 0)[0].shape[0]
        num_pos = np.where(Y_copy == 1)[0].shape[0]
        keep_num_pos = int((imratio / (1 - imratio)) * num_neg)
        neg_id_list = np.where(Y_copy == 0)[0]
        pos_id_list = np.where(Y_copy == 1)[0][:keep_num_pos]
        X_copy = X_copy[neg_id_list.tolist() + pos_id_list.tolist()]
        Y_copy = Y_copy[neg_id_list.tolist() + pos_id_list.tolist()]
        # Y_copy[Y_copy==0] = 0

    if shuffle:
        # do shuffle in case batch prediction error
        id_list = list(range(X_copy.shape[0]))
        np.random.seed(random_seed)
        np.random.shuffle(id_list)
        X_copy = X_copy[id_list]
        Y_copy = Y_copy[id_list]

    num_samples = len(X_copy)
    pos_count = np.count_nonzero(Y_copy == 1)
    neg_count = np.count_nonzero(Y_copy == 0)
    pos_ratio = pos_count / (pos_count + neg_count)
    #print('NUM_SAMPLES: [%d], POS:NEG: [%d : %d], POS_RATIO: %.4f' % (num_samples, pos_count, neg_count, pos_ratio))

    return X_copy, Y_copy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--i', default='raw/', help='input directory')
    parser.add_argument('--o', default='mnist_imbalanced_permutations.pt', help='output file')
    parser.add_argument('--seed', default=123, type=int, help='random seed')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    x_tr, y_tr = torch.load(os.path.join(args.i, 'mnist_train.pt'))
    x_te, y_te = torch.load(os.path.join(args.i, 'mnist_test.pt'))

    x_tr, y_tr = ImbalanceGenerator(x_tr.numpy(), y_tr.numpy(), is_balanced=False, imratio=0.1)
    x_te, y_te = ImbalanceGenerator(x_te.numpy(), y_te.numpy(), is_balanced=True)

    x_tr, y_tr = torch.tensor(x_tr), torch.tensor(y_tr)
    x_te, y_te = torch.tensor(x_te), torch.tensor(y_te)

    torch.save((x_tr, y_tr), 'mnist_imbalanced_train.pt')
    torch.save((x_te, y_te), 'mnist_imbalanced_test.pt')



