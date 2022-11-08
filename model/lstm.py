import torch
import numpy as np
import torch.nn as nn

'''
https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM
http://colah.github.io/posts/2015-08-Understanding-LSTMs/
'''


class LSTM(nn.Module):
    def __init__(self, cfg):
        super(LSTM, self).__init__()
        self.input_size = cfg['Global']['input_size']
        self.output_size = cfg['Global']['output_size']
        self.hidden_size = cfg['Architecture']['hidden_size']
        self.num_layers = cfg['Architecture']['num_layers']
        self.bidirectional = cfg['Architecture']['bidirectional']
        self.D = 2 if self.bidirectional else 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True,
                            num_layers=self.num_layers, bidirectional=self.bidirectional)
        self.fc = nn.Linear(self.hidden_size * self.D, self.output_size)

    def forward(self, x):
        batch_size, seq_len, input_size = x.shape
        h_0 = torch.randn(self.num_layers * self.D, batch_size, self.hidden_size).to(self.device)
        c_0 = torch.randn(self.num_layers * self.D, batch_size, self.hidden_size).to(self.device)
        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        output = self.fc(out[:, -1, :]).squeeze(0)  # 因为有max_seq_len个时态，所以取最后一个时态即-1层
        return output





