from tqdm import tqdm
from data_loader import StockDataset, StockDataloader
from typing import Tuple
import pandas as pd
import numpy as np
import os


def generate_train_and_test(root_dir: str) -> None:
    """
    Preprocess data to generate training and testing data
    :param root_dir: Directory containing raw data
    :return: None
    """
    if not os.path.exists('data') or not os.path.exists('original_data'):
        os.makedirs('data/train')
        os.makedirs('data/test')
        os.makedirs('original_data/train')
        os.makedirs('original_data/test')
    else:
        return

    symbol_list = pd.read_csv(f'{root_dir}/symbols_valid_meta.csv',
                              delimiter=',',
                              usecols=['Nasdaq Traded',
                                       'Symbol',
                                       'Security Name',
                                       'Listing Exchange',
                                       'Market Category',
                                       'ETF',
                                       'Round Lot Size',
                                       'Test Issue',
                                       'Financial Status',
                                       'CQS Symbol',
                                       'NASDAQ Symbol',
                                       'NextShares'])

    existing_symbols = []
    for symbol in tqdm(symbol_list['Symbol']):
        stock = None
        if os.path.exists(f'{root_dir}/stocks/{symbol}.csv'):
            stock = pd.read_csv(f'{root_dir}/stocks/{symbol}.csv',
                                delimiter=',',
                                usecols=['Date',
                                         'Open',
                                         'High',
                                         'Low',
                                         'Close',
                                         'Adj Close',
                                         'Volume'])
        elif os.path.exists(f'{root_dir}/etfs/{symbol}.csv'):
            stock = pd.read_csv(f'{root_dir}/etfs/{symbol}.csv',
                                delimiter=',',
                                usecols=['Date',
                                         'Open',
                                         'High',
                                         'Low',
                                         'Close',
                                         'Adj Close',
                                         'Volume'])

        if stock is not None:
            existing_symbols.append(symbol)

            stock.sort_values('Date', inplace=True)
            stock.drop(columns=['Date', 'Adj Close'], inplace=True)

            # Deal with invalid values
            stock.replace([np.inf, -np.inf], np.nan, inplace=True)
            stock.dropna(how='any', axis=0, inplace=True)
            stock['High'].replace(to_replace=0, method='ffill', inplace=True)
            stock['Open'] = np.where(stock['Open'] == 0,
                                     (stock['High'] + stock['Low']) / 2.0,
                                     stock['Open'])
            stock['Close'] = np.where(stock['Close'] == 0,
                                      stock['Open'],
                                      stock['Close'])
            stock['Low'] = np.where(stock['Low'] == 0,
                                    stock['Open'] * 2.1 - stock['High'],
                                    stock['Low'])
            stock['Volume'].replace(to_replace=0, method='ffill', inplace=True)
            stock['Volume'].replace(to_replace=0, value=1, inplace=True)

            # Store original data
            last_20_percent = -int(0.2 * len(stock))

            stock_train = stock.iloc[:last_20_percent, :].copy()
            stock_train.to_csv(f'original_data/train/{symbol}.csv', index=False)
            del stock_train

            stock_test = stock.iloc[last_20_percent:, :].copy()
            stock_test.to_csv(f'original_data/test/{symbol}.csv', index=False)
            del stock_test

            # Convert data to percentage change
            stock['Open'] = stock['Open'].pct_change()
            stock['High'] = stock['High'].pct_change()
            stock['Low'] = stock['Low'].pct_change()
            stock['Close'] = stock['Close'].pct_change()
            stock['Volume'] = stock['Volume'].pct_change()
            stock.replace([np.inf, -np.inf], np.nan, inplace=True)
            stock.dropna(how='any', axis=0, inplace=True)

            # Store converted data
            stock_train = stock.iloc[:last_20_percent, :].copy()
            stock_train.to_csv(f'data/train/{symbol}.csv', index=False)
            del stock_train

            stock_test = stock.iloc[last_20_percent:, :].copy()
            stock_test.to_csv(f'data/test/{symbol}.csv', index=False)
            del stock_test

            del stock
    existing_symbols = pd.DataFrame(existing_symbols, columns=['Symbol'])
    existing_symbols.to_csv(f'data/symbols.csv', index=False)


def get_data_loaders(batch_size: int, seq_len: int) -> Tuple[StockDataloader, StockDataloader]:
    """
    Get training and testing data loaders
    :param batch_size: Batch size
    :param seq_len: Sequence length
    :return: Training data loader and testing data loader
    """
    symbols = pd.read_csv(f'data/symbols.csv',
                          delimiter=',',
                          usecols=['Symbol'])
    symbols = symbols['Symbol'].values.tolist()

    train_datasets = [StockDataset(mode='train', symbol=symbol, seq_len=seq_len) for symbol in symbols]
    test_datasets = [StockDataset(mode='test', symbol=symbol, seq_len=seq_len) for symbol in symbols]

    # Find train datasets with insufficient sequence
    train_insufficient = []
    for idx, dataset in enumerate(train_datasets):
        if len(dataset) == 0:
            train_insufficient.append(idx)

    # Find test datasets with insufficient sequence
    test_insufficient = []
    for idx, dataset in enumerate(test_datasets):
        if len(dataset) == 0:
            test_insufficient.append(idx)

    # Find common insufficient list
    insufficient = train_insufficient + list(set(test_insufficient) - set(train_insufficient))
    insufficient.sort()

    # Remove stocks with insufficient length
    for idx in reversed(insufficient):
        del symbols[idx]
        del train_datasets[idx]
        del test_datasets[idx]

    return StockDataloader(symbols=symbols, datasets=train_datasets, batch_size=batch_size), StockDataloader(
        symbols=symbols, datasets=test_datasets, batch_size=batch_size)
