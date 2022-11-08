import torch.optim as optim


class Adam:
    def __init__(self, model, lr=0.01):
        self.model = model
        self.lr = lr

    def __call__(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer


class SGD:
    def __init__(self, model, lr=0.01, momentum=0.9):
        self.model = model
        self.lr = lr
        self.momentum = momentum

    def __call__(self):
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        return optimizer
