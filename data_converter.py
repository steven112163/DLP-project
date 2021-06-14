import pandas as pd
import os


def generate_train_and_test(root_dir: str) -> None:
    """
    Preprocess data to generate training and testing data
    :param root_dir: Directory containing raw data
    :return: None
    """
    if not os.path.exists('data'):
        os.makedirs('data/train')
        os.makedirs('data/test')
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

    exists_symbols = []
    for symbol in symbol_list['Symbol']:
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
            exists_symbols.append(symbol)

            stock.sort_values('Date', inplace=True)

            stock['Open'] = stock['Open'].pct_change()
            stock['High'] = stock['High'].pct_change()
            stock['Low'] = stock['Low'].pct_change()
            stock['Close'] = stock['Close'].pct_change()
            stock['Volume'] = stock['Volume'].pct_change()

            stock.dropna(how='any', axis=0, inplace=True)
            stock.replace(to_replace=0, method='ffill', inplace=True)

            min_value = min(stock[['Open', 'High', 'Low', 'Close']].min(axis=0))
            max_value = max(stock[['Open', 'High', 'Low', 'Close']].max(axis=0))

            stock['Open'] = (stock['Open'] - min_value)/(max_value - min_value)
            stock['High'] = (stock['High'] - min_value)/(max_value - min_value)
            stock['Low'] = (stock['Low'] - min_value)/(max_value - min_value)
            stock['Close'] = (stock['Close'] - min_value)/(max_value - min_value)
            
            min_volume = min(stock[['Volume']].min(axis=0))
            max_volume = max(stock[['Volume']].max(axis=0))
            stock['Volume'] = (stock['Volume'] - min_volume)/(max_volume - min_volume)
                    
            last_20_percent = -int(0.2 * len(stock))

            stock_train = stock.iloc[:last_20_percent, :].copy()
            stock_train.drop(columns=['Date', 'Adj Close'], inplace=True)
            stock_train.to_csv(f'data/train/{symbol}.csv', index=False)

            stock_test = stock.iloc[last_20_percent:, :].copy()
            stock_test.drop(columns=['Date', 'Adj Close'], inplace=True)
            stock_test.to_csv(f'data/test/{symbol}.csv', index=False)

    exists_symbols = pd.DataFrame(exists_symbols, columns=['Symbol'])
    exists_symbols.to_csv(f'data/symbols.csv', index=False)
