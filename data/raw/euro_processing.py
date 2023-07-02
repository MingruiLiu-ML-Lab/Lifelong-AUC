from skimage import io
import torch
import torchvision
import torch.nn.init
from torchvision import transforms
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image

def iloader(path):
    #image = np.asarray((io.imread(path)),dtype='float32')
    #image = io.imread(path)
    img = Image.open(path)
    # return image.transpose(2,0,1)
    return img

data_transforms=transforms.Compose([transforms.Resize([32, 32]),
                                    transforms.ToTensor(),
                                    # transforms.Normalize((0.3442, 0.3801, 0.4076), (0.2034, 0.1365, 0.1148))
                                 ])


if __name__ == '__main__':

    root = '2750/'
    data = torchvision.datasets.DatasetFolder(root=root, loader=iloader, transform=data_transforms, extensions='jpg')
    train_set, test_set, train_target, test_target = train_test_split(data, data.targets, test_size=0.2, stratify=data.targets)
    train_data = np.zeros((len(train_target), 3, 32, 32))
    test_data = np.zeros((len(test_target), 3, 32, 32))
    for i in range(len(train_target)):
        train_data[i] = train_set[i][0]
    for i in range(len(test_target)):
        test_data[i] = test_set[i][0]
    train_data = torch.Tensor(train_data)/train_data.max()
    test_data = torch.Tensor(test_data)/test_data.max()
    train_target = torch.LongTensor(train_target)
    test_target = torch.LongTensor(test_target)
    torch.save((train_data, train_target, test_data, test_target),'eurosat.pt')

