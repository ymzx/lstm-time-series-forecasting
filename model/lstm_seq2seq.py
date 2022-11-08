import torch
import torch.nn as nn


'''
采用编码-解码结构，实现'N V N'输入和输出，
'''


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_size = cfg['Global']['input_size']
        self.output_size = cfg['Global']['output_size']
        self.hidden_size = cfg['Architecture']['hidden_size']
        self.num_layers = cfg['Architecture']['num_layers']
        self.bidirectional = cfg['Architecture']['bidirectional']
        self.D = 2 if self.bidirectional else 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=self.bidirectional)

    def forward(self, x):
        batch_size, seq_len, input_size = x.shape
        h_0 = torch.randn(self.num_layers * self.D, batch_size, self.hidden_size).to(self.device)
        c_0 = torch.randn(self.num_layers * self.D, batch_size, self.hidden_size).to(self.device)
        output, (h, c) = self.lstm(x, (h_0, c_0))
        return h, c


class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_size = cfg['Global']['input_size']
        self.output_size = cfg['Global']['output_size']
        self.hidden_size = cfg['Architecture']['hidden_size']
        self.num_layers = cfg['Architecture']['num_layers']
        self.bidirectional = cfg['Architecture']['bidirectional']
        self.D = 2 if self.bidirectional else 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=self.bidirectional)
        self.linear = nn.Linear(self.hidden_size*self.D, self.input_size)

    def forward(self, x, h, c):
        if x.dim == 3:
            batch_size, seq_len, input_size = x.shape
        else:
            batch_size, seq_len = x.shape
        input_seq = x.view(batch_size, 1, self.input_size)
        output, (h, c) = self.lstm(input_seq, (h, c))
        pred = self.linear(output)
        pred = pred[:, -1, :]
        return pred, h, c


class Seq2Seq(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.output_size = cfg['Global']['output_size']
        self.input_size = cfg['Global']['input_size']  # 数据加载时，赋值到cfg
        self.Encoder = Encoder(cfg)
        self.Decoder = Decoder(cfg)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.linear = nn.Linear(self.input_size, self.output_size)
        self.linear_out = nn.Linear(self.input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, seq_len, input_size = x.shape
        h, c = self.Encoder(x)
        outputs = torch.zeros(batch_size, self.output_size).to(self.device)
        _input = torch.zeros(batch_size, input_size).to(self.device)
        for i in range(self.output_size):
            output, h, c = self.Decoder(_input, h, c)
            _input = output
            out = self.linear_out(output)
            out = self.sigmoid(out)
            outputs[:, i] = out[:, 0]
        return outputs
