import torch
import torch.nn.functional as F


class AUCMLoss(torch.nn.Module):

    def __init__(self, margin=1.0, imratio=None, device=None):
        super(AUCMLoss, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.margin = margin
        self.p = imratio

        self.a = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.b = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.alpha = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)

    def forward(self, y_pred, y_true):
        # y_pred = torch.softmax(y_pred, dim=1)
        #y_pred = y_pred[:, -1] # [0.1 (cls=0), 0.9(cls=1)] => [0.9] only positive score
        y_pred = torch.sigmoid(y_pred)
        if self.p is None:
            self.p = (y_true == 1).float().sum() / y_true.shape[0]

        y_pred = y_pred.reshape(-1, 1)  # be carefull about these shapes
        y_true = y_true.reshape(-1, 1)
        loss = (1 - self.p) * torch.mean((y_pred - self.a) ** 2 * (1 == y_true).float()) + \
               self.p * torch.mean((y_pred - self.b) ** 2 * (0 == y_true).float()) + \
               2 * self.alpha * (self.p * (1 - self.p) * self.margin + \
                                 torch.mean((self.p * y_pred * (0 == y_true).float() - (1 - self.p) * y_pred * (
                                             1 == y_true).float()))) - \
               self.p * (1 - self.p) * self.alpha ** 2
        return loss


class AUCM_MultiLabel(torch.nn.Module):

    def __init__(self, margin=1.0, imratio=[0.1], num_classes=10, device=None):
        super(AUCM_MultiLabel, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.margin = margin
        self.p = torch.FloatTensor(imratio).to(self.device)
        self.num_classes = num_classes
        assert len(imratio) == num_classes, 'Length of imratio needs to be same as num_classes!'
        self.a = torch.zeros(num_classes, dtype=torch.float32, device="cuda", requires_grad=True).to(self.device)
        self.b = torch.zeros(num_classes, dtype=torch.float32, device="cuda", requires_grad=True).to(self.device)
        self.alpha = torch.zeros(num_classes, dtype=torch.float32, device="cuda", requires_grad=True).to(self.device)

    @property
    def get_a(self):
        return self.a.mean()

    @property
    def get_b(self):
        return self.b.mean()

    @property
    def get_alpha(self):
        return self.alpha.mean()

    def forward(self, y_pred, y_true):
        one_hot = torch.zeros(y_pred.size(), device=y_true.device)
        one_hot.scatter_(1, y_true.view(-1, 1).long(), 1)
        y_true = one_hot
        # matrix operate faster
        y_pred_i = y_pred
        y_true_i = y_true
        loss = (1 - self.p) * torch.mean((y_pred_i - self.a) ** 2 * (1 == y_true_i).float()) + \
               self.p * torch.mean((y_pred_i - self.b) ** 2 * (0 == y_true_i).float()) + \
               2 * self.alpha * (self.p * (1 - self.p) + \
                                      torch.mean((self.p * y_pred_i * (0 == y_true_i).float() - (
                                              1 - self.p) * y_pred_i * (1 == y_true_i).float()))) - \
               self.p * (1 - self.p) * self.alpha ** 2
        total_loss = sum(loss)
        return total_loss
