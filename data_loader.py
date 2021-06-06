from torch.utils import data
import torch
import pandas as pd

def GetData(mode, symbol):
    if mode == 'train':
        data = pd.read_csv(f'data/train/{symbol}.csv')
    else:
        data = pd.read_csv(f'data/test/{symbol}.csv')

    return data

class DataLoader(data.Dataset):
    def __init__(self, root, mode, symbol, seq_len=128):
        # TODO
        self.csv_data = GetData(mode, symbol)
        self.datatensor = torch.tensor(self.csv_data['Open', 'High', 'Low', 'Close', 'Volume'].values())
        self.seq_len = seq_len
        self.data = []
        self.close = []

    def __len__(self):
        """
        Number of data
        :return: number of data
        """
        # TODO
        return self.csv_data.shape[0] - self.seq_len + 1
    
    def __getitem__(self, index):
        """
        Get current data
        :param index: index of training/testing data
        :return: data
        """
        # TODO
        for i in range(index, index + self.seq_len):
            self.data.append(self.datatensor[i-self.seq_len:i])
            self.close.append(self.datatensor[:, 3][i])

        return torch.FloatTensor(self.data), torch.FloatTensor(self.close)