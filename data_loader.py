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
        self.data_tensor = torch.FloatTensor(csv_data.values.tolist())
        self.seq_len = seq_len

    def __len__(self) -> int:
        """
        Number of data
        :return: Number of data
        """
        length = self.data_tensor.size(0) - self.seq_len
        return 0 if length < 0 else length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        """
        Get current data
        :param index: Index of training/testing data
        :return: Data
        """
        sequence = self.data_tensor[index:index + self.seq_len]
        close = self.data_tensor[index + self.seq_len, 3]

        return sequence, close


class StockDataloader:
    def __init__(self, mode: str, batch_size: int, seq_len: int):
        super(StockDataloader, self).__init__()

        self.symbols = pd.read_csv(f'data/symbols.csv',
                                   delimiter=',',
                                   usecols=['Symbol'])

        self.stocks = []
        for symbol in self.symbols['Symbol']:
            dataset = StockDataset(mode=mode, symbol=symbol, seq_len=seq_len)
            if len(dataset) > 0:
                # Only get the stock with enough length
                self.stocks.append(DataLoader(dataset, batch_size=batch_size))

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
