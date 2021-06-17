from torch.utils.data import Dataset, DataLoader
from typing import Tuple
import torch
import pandas as pd


class StockDataset(Dataset):
    def __init__(self, mode: str, symbol: str, seq_len: int):
        super(StockDataset, self).__init__()

        csv_data = pd.read_csv(f'data/{mode}/{symbol}.csv',
                               delimiter=',',
                               usecols=['Open', 'High', 'Low', 'Close', 'Volume'])
        self.data_tensor = torch.tensor(csv_data.values.tolist())
        self.seq_len = seq_len

    def __len__(self) -> int:
        """
        Number of data
        :return: Number of data
        """
        return self.data_tensor.size(0) - self.seq_len + 1

    def __getitem__(self, index: int) -> Tuple[torch.FloatTensor, ...]:
        """
        Get current data
        :param index: Index of training/testing data
        :return: Data
        """
        # TODO
        sequence, close = [], []
        for idx in range(index, index + self.seq_len):
            sequence.append(self.data_tensor[idx - self.seq_len:idx])
            close.append(self.data_tensor[:, 3][idx])

        return torch.FloatTensor(sequence), torch.FloatTensor(close)


class StockDataloader:
    def __init__(self, mode: str, batch_size: int, seq_len: int):
        super(StockDataloader, self).__init__()

        self.symbols = pd.read_csv(f'data/symbols.csv',
                                   delimiter=',',
                                   usecols=['Symbol'])
        self.stocks = [DataLoader(StockDataset(mode=mode, symbol=symbol, seq_len=seq_len),
                                  batch_size=batch_size)
                       for symbol in self.symbols['Symbol']]

    def __len__(self) -> int:
        """
        Number of dataset
        :return: Number of dataset
        """
        return len(self.stocks)

    def __getitem__(self, index: int) -> Tuple[str, DataLoader]:
        """
        Get current stock data loader with its symbol
        :param index: Index of stocks
        :return: Current stock data loader with its symbol
        """
        return self.symbols['Symbol'][index], self.stocks[index]
