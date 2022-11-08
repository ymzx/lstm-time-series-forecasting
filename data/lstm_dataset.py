from torch.utils.data import Dataset, DataLoader
import torch


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


class LSTMDataSet:

    def __init__(self, cfg):
        self.cfg = cfg
        self.lookback = cfg['Global']['seq_len']
        self.pre_steps = cfg['Global']['output_size']
        self.batch_size = cfg['Train']['loader']['batch_size']
        self.num_workers = cfg['Train']['loader']['num_workers']
        self.train_ratio = cfg['Train']['loader']['train_ratio']
        self.shuffle = cfg['Train']['loader']['shuffle']

    def __call__(self, data):
        self.data = data
        data_list = self.data.values.tolist()
        seq, input_size = [], 0
        for i in range(0, len(data_list) - self.lookback - self.pre_steps, self.pre_steps):
            train_seq = []
            train_label = []
            for j in range(i, i + self.lookback):
                train_seq.append(data_list[i])
                input_size = len(data_list[i])
            for j in range(i + self.lookback, i + self.lookback + self.pre_steps):
                train_label.append(data_list[j][-1])
            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label)
            seq.append((train_seq, train_label))
        # 写入input_size
        self.cfg['Global']['input_size'] = input_size
        # 切分训练测试集
        train_data = seq[0:int(len(seq) * self.train_ratio)]
        test_data = seq[int(len(seq) * self.train_ratio):len(seq)]
        train_len = int(len(train_data) / self.batch_size) * self.batch_size
        test_len = int(len(test_data) / self.batch_size) * self.batch_size
        train_data, test_data = train_data[:train_len], test_data[:test_len]
        train, test = MyDataset(train_data), MyDataset(test_data)
        train_loader = DataLoader(dataset=train, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
        test_loader = DataLoader(dataset=test, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
        return train_loader, test_loader


if __name__ == '__main__':
    pass

