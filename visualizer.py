from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_loss(losses: Tuple[List[float], ...], epoch: int, label=List[str]) -> None:
    """
    Plot losses
    :param losses: Losses
    :param epoch: Current epoch
    :param label: List of labels
    :return: None
    """
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    for idx, loss in enumerate(losses):
        plt.plot(range(epoch + 1), loss[:epoch + 1], label=label[idx])
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./figures/loss.png')


def plot_predicted_results(train_predictions: Dict[str, List[float]],
                           test_predictions: Dict[str, List[float]],
                           seq_len: int) -> None:
    """
    Plot predicted results
    :param train_predictions: Dictionary of 5 lists of training predictions
    :param test_predictions: Dictionary of 5 lists of testing predictions
    :param seq_len: Sequence length
    :return: None
    """
    for symbol in train_predictions.keys():
        plt.clf()
        fig = plt.figure()

        # Plot training data and predictions
        train_data = pd.read_csv(f'data/train/{symbol}.csv',
                                 delimiter=',',
                                 usecols=['Open', 'High', 'Low', 'Close', 'Volume'])
        ax = fig.add_subplot(211)
        ax.set_title('Training Data')
        ax.set_xlabel('Date')
        ax.set_ylabel('Closing Returns')
        ax.plot(train_data['Close'], label='Closing Returns')
        ax.plot(range(seq_len, len(train_predictions[symbol]) + seq_len),
                train_predictions[symbol],
                linewidth=3,
                label='Predicted Closing Returns')
        ax.legend(loc='best')
        del train_data

        test_data = pd.read_csv(f'data/test/{symbol}.csv',
                                delimiter=',',
                                usecols=['Open', 'High', 'Low', 'Close', 'Volume'])
        ax = fig.add_subplot(212)
        ax.set_title('Testing Data')
        ax.set_xlabel('Date')
        ax.set_ylabel('Closing Returns')
        ax.plot(test_data['Close'], label='Closing Returns')
        ax.plot(range(seq_len, len(test_predictions[symbol]) + seq_len),
                test_predictions[symbol],
                label='Predicted Closing Returns')
        ax.legend(loc='best')
        del test_data

        plt.tight_layout()
        plt.savefig(f'./figures/train/{symbol}_prediction.png')


def plot_inference_results(predictions: Dict[str, Dict[str, List[float]]],
                           seq_len: int):
    """
    Plot inference results
    :param predictions: Dictionary of 10 lists of inferring train/test predictions
    :param seq_len: Sequence length
    :return: None
    """
    for symbol in predictions.keys():
        plt.clf()
        fig = plt.figure()

        results = f'MSE, MAE, and MAPE of stock {symbol} (training|testing):   '

        # Plot predicted stock price (training data)
        ax = fig.add_subplot(211)
        results += plot_stock(symbol=symbol,
                              ax=ax,
                              predictions=predictions,
                              seq_len=seq_len,
                              mode='train')
        results += ' | '

        # Plot predicted stock price (testing data)
        ax = fig.add_subplot(212)
        results += plot_stock(symbol=symbol,
                              ax=ax,
                              predictions=predictions,
                              seq_len=seq_len,
                              mode='test')

        plt.tight_layout()
        plt.savefig(f'./figures/inference/{symbol}_prediction.png')

        print(results)


def plot_stock(symbol: str,
               ax,
               predictions: Dict[str, Dict[str, List[float]]],
               seq_len: int,
               mode: str) -> str:
    """
    Plot stock price
    :param symbol: Target symbol
    :param ax: Axis
    :param predictions: Dictionary of 10 lists of inferring train/test predictions
    :param seq_len: Sequence length
    :param mode: Train or test
    :return: String of MSE, MAE, and MAPE
    """
    stock_data = pd.read_csv(f'original_data/{mode}/{symbol}.csv',
                             delimiter=',',
                             usecols=['Open', 'High', 'Low', 'Close', 'Volume'])
    ax.set_title(f'{"Training" if mode == "train" else "Testing"} Data')
    ax.set_xlabel('Date')
    ax.set_ylabel('Closing Returns')
    ax.plot(stock_data['Close'], label='Closing Returns')
    predictions[symbol][f'{mode}'] = np.add(predictions[symbol][f'{mode}'], 1.0)
    original_data = np.array(stock_data['Close'][-len(predictions[symbol][f'{mode}']) - 1:-1].values.tolist())
    predicted = np.multiply(original_data, predictions[symbol][f'{mode}'])
    ax.plot(range(seq_len, len(predicted) + seq_len),
            predicted,
            label='Predicted Closing Returns')
    ax.legend(loc='best')
    del stock_data

    # Print MSE, MAE, and MAPE of the given stock
    predicted = np.array(predicted)
    difference = original_data - predicted
    mse = np.sum(difference ** 2) / len(predicted)
    mae = np.sum(np.abs(difference)) / len(predicted)
    mape = np.sum(np.abs(difference) / original_data) / len(predicted) * 100

    return f'({mse:.4f}, {mae:.4f}, {mape:.4f} %)'
