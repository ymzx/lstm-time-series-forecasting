import torch
import torch.nn as nn
import torch.nn.functional as F

'''
https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
'''


class CNNLSTM(nn.Module):
    def __init__(self, cfg):
        super(CNNLSTM, self).__init__()
        self.input_size = cfg['Global']['input_size']
        self.output_size = cfg['Global']['output_size']
        self.seq_len = cfg['Global']['seq_len']
        self.kernel_size = cfg['Architecture']['kernel_size']
        self.stride = cfg['Architecture']['stride']
        self.hidden_size = cfg['Architecture']['hidden_size']
        self.num_layers = cfg['Architecture']['num_layers']
        self.dilation = cfg['Architecture']['dilation']

        self.conv1 = nn.Conv1d(self.input_size, 64, kernel_size=self.kernel_size, stride=self.stride,
                               dilation=self.dilation)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=self.kernel_size, stride=self.stride, padding=1)
        self.batch1 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=self.kernel_size, stride=self.stride, padding=1)
        self.batch2 = nn.BatchNorm1d(32)
        # self.Lout = round((self.seq_len - self.dilation * (self.kernel_size - 1) - 1) / self.stride + 1)
        self.LSTM = nn.LSTM(input_size=32, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        # 数据输入[batce_size,seq_length,input_size], 卷积层输入[batch_size,input_size,seq_length]
        x = x.permute(0, 2, 1)
        x = F.selu(self.conv1(x))
        x = self.conv2(x)
        x = F.selu(self.batch1(x))
        x = self.conv3(x)
        x = F.selu(self.batch2(x))
        # LSTM层输入 (batch_size,seq_length,out_channels)
        x = x.permute(0, 2, 1)
        x, h = self.LSTM(x)
        output = self.fc(x[:, -1, :])
        return output
