import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from isicDataset import ISICDataset

torch.manual_seed(0)

labels = pd.read_csv('ISIC/ISIC_2019_Training_GroundTruth.csv')
labels = labels.sample(frac=1).reset_index(drop=True)
train_set = labels.iloc[5331:, :]
eval_set = labels.iloc[:5331, :]

eval_set.to_csv('ISIC/validation.csv', index=False)
train_set.to_csv('ISIC/training.csv', index=False)

train_path = 'ISIC/training.csv'
eval_path = 'ISIC/validation.csv'


train_img_path = 'ISIC/ISIC_2019_Training_Input'




data_transforms = {
    'train': transforms.Compose([transforms.Resize([32, 32]),
                                transforms.ToTensor(),
                                transforms.Normalize((0.6678, 0.5298, 0.5245), (0.1333, 0.1476, 0.1590))
                                 ]),
    'val': transforms.Compose([transforms.Resize([32, 32]),
                               transforms.ToTensor(),
                               transforms.Normalize((0.6678, 0.5298, 0.5245), (0.1333, 0.1476, 0.1590))
                               ]),
}

data_set_train = ISICDataset(
    train_img_path, transform=data_transforms['train'],
    csv_path=train_path)

data_set_val = ISICDataset(
    img_path=train_img_path, transform=data_transforms['val'],
    csv_path=eval_path)


if __name__ == '__main__':

    train_num = data_set_train.__len__()
    test_num = data_set_val.__len__()
    train_data = np.zeros((train_num, 3, 32, 32))
    test_data = np.zeros((test_num, 3, 32, 32))
    train_label = np.zeros(train_num)
    test_label = np.zeros(test_num)
    for index in range(train_num):
        train_data[index], train_label[index] = data_set_train.__getitem__(index)
    for index in range(test_num):
        test_data[index], test_label[index] = data_set_val.__getitem__(index)
    torch.save((torch.from_numpy(train_data), torch.LongTensor(train_label), torch.from_numpy(test_data), torch.LongTensor(test_label)),'ISIC_data.pt')
