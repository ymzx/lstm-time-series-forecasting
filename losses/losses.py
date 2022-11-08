import torch.nn as nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MSELoss(nn.MSELoss):
    def __int__(self):
        super(MSELoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        se = torch.pow((input - target), 2)
        mse = torch.mean(se)
        return mse


class MAELoss(nn.MSELoss):
    def __int__(self):
        super(MAELoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        se = torch.abs(input - target)
        mse = torch.mean(se)
        return mse


class MSEZeroMaskLoss(nn.MSELoss):
    def __int__(self):
        super(MSEZeroMaskLoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        se = torch.pow((input - target), 2)
        se[target == 0] = 0
        mse = torch.mean(se)
        return mse


class MAEZeroMaskLoss(nn.MSELoss):
    def __int__(self):
        super(MAEZeroMaskLoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        se = torch.abs(input - target)
        se[target == 0] = 0
        mse = torch.mean(se)
        return mse


class WMAELoss(nn.MSELoss):
    def __int__(self):
        super(WMAELoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ae = torch.abs(input - target)
        wae = torch.mul(target.abs(), ae)
        loss = torch.mean(wae)
        return loss


class WMSELoss(nn.MSELoss):
    def __int__(self):
        super(WMSELoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ae = torch.pow((input - target), 2)
        wae = torch.mul(target.abs()+0.02, ae)
        loss = torch.mean(wae)
        return loss


# class CEL:
#
#     def __init__(self):
#         pass
#
#     def __call__(self, *args, **kwargs):
#         loss_func = nn.CrossEntropyLoss().to(device)
#         return loss_func
#
#
# class WMAPELoss(nn.MSELoss): # 注意继承 nn.Module
#     def __init__(self):
#         super(WMAPELoss, self).__init__()
#
#     def forward(self, y_hat, y):
#         diff = (y_hat-y).abs().sum()
#         denominator = y.abs().sum()
#         if denominator == 0:
#             wmape = torch.scalar_tensor(0, requires_grad=True, device=device)
#         else:
#             wmape = torch.div(diff, denominator)
#         return wmape  # 注意最后只能返回Tensor值，且带梯度，即 loss.requires_grad == True
#
#
# class WMAPEZeroMaskLoss(nn.Module): # 注意继承 nn.Module
#     def __init__(self):
#         super(WMAPEZeroMaskLoss, self).__init__()
#
#     def forward(self, y_hat, y):
#         diff = (y_hat-y).abs()
#         diff[y == 0] = 0
#         diff = diff.mean()
#         denominator = y.abs().mean()
#         wmape = diff/denominator
#         return wmape
