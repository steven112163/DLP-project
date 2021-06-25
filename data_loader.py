from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List
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
        length = self.data_tensor.size(0) - self.seq_len - 1
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
    def __init__(self, symbols: List[str], datasets: List[StockDataset], batch_size: int):
        super(StockDataloader, self).__init__()

        self.symbols = symbols
        self.stocks = [DataLoader(dataset, batch_size=batch_size) for dataset in datasets]

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
        return self.symbols[index], self.stocks[index]
