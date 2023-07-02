import torch
import copy


class PESG(torch.optim.Optimizer):


    def __init__(self,
                 model,
                 a=None,
                 b=None,
                 alpha=None,
                 imratio=0.1,
                 margin=1.0,
                 lr=0.1,
                 gamma=500,
                 clip_value=1.0,
                 weight_decay=1e-5,
                 device=None,
                 **kwargs):

        assert a is not None, 'Found no variable a!'
        assert b is not None, 'Found no variable b!'
        assert alpha is not None, 'Found no variable alpha!'

        self.p = imratio
        self.margin = margin
        self.model = model

        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.lr = lr
        self.gamma = gamma
        self.clip_value = clip_value
        self.weight_decay = weight_decay

        self.a = a
        self.b = b
        self.alpha = alpha

        self.T = 0
        self.step_counts = 0

        def get_parameters(params):
            for p in params:
                yield p

        if model is None:
            self.params = get_parameters([a, b])
        else:
            self.params = get_parameters(list(model.parameters()) + [a, b])

        self.defaults = dict(lr=self.lr,
                             margin=margin,
                             gamma=gamma,
                             p=imratio,
                             a=self.a,
                             b=self.b,
                             alpha=self.alpha,
                             clip_value=clip_value,
                             weight_decay=weight_decay,
                             )

        super(PESG, self).__init__(self.params, self.defaults)

    def init_model_ref(self):
        self.model_ref = []
        for var in list(self.model.parameters()) + [self.a, self.b]:
            self.model_ref.append(torch.empty(var.shape).normal_(mean=0, std=0.01).to(self.device))
        return self.model_ref

    def init_model_acc(self):
        self.model_acc = []
        for var in list(self.model.parameters()) + [self.a, self.b]:
            self.model_acc.append(
                torch.zeros(var.shape, dtype=torch.float32, device=self.device, requires_grad=False).to(self.device))
        return self.model_acc

    @property
    def optim_steps(self):
        return self.step_counts

    @property
    def get_params(self):
        return self.params

    def update_lr(self, lr):
        self.param_groups[0]['lr'] = lr

    @torch.no_grad()
    def step(self):
        """Performs a single optimization step.
        """
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            clip_value = group['clip_value']
            self.lr = group['lr']

            p = group['p']
            gamma = group['gamma']
            m = group['margin']

            a = group['a']
            b = group['b']
            alpha = group['alpha']

            # updates
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                p.data = p.data - group['lr'] * torch.clamp(p.grad.data, -clip_value, clip_value) # (w, a, b) gradient descend

            # (alpha) gradient ascend
            alpha.data = alpha.data + group['lr'] * (2 * (m + b.data - a.data) - 2 * alpha.data)
            alpha.data = torch.clamp(alpha.data, 0, 999)

        self.T += 1
        self.step_counts += 1

    def zero_grad(self):
        if self.model:
            self.model.zero_grad()
        self.a.grad = None
        self.b.grad = None
        self.alpha.grad = None

    def update_regularizer(self, decay_factor=None):
        if decay_factor != None:
            self.param_groups[0]['lr'] = self.param_groups[0]['lr'] / decay_factor
            print('Reducing learning rate to %.5f @ T=%s!' % (self.param_groups[0]['lr'], self.T))
        print('Updating regularizer @ T=%s!' % (self.T))
        for i, param in enumerate(self.model_ref):
            self.model_ref[i].data = self.model_acc[i].data / self.T
        for i, param in enumerate(self.model_acc):
            self.model_acc[i].data = torch.zeros(param.shape, dtype=torch.float32, device=self.device,
                                                 requires_grad=False).to(self.device)
        self.T = 0
