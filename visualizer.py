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
    for loss in losses:
        plt.plot(range(epoch + 1), loss[:epoch + 1], label=label)
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


def plot_inference_results(predictions: Dict[str, List[float]],
                           seq_len: int):
    """
    Plot inference results
    :param predictions: Dictionary of 10 lists of inferring predictions
    :param seq_len: Sequence length
    :return: None
    """
    for symbol in predictions.keys():
        plt.clf()
        fig = plt.figure()

        # Plot predicted percentage
        test_data = pd.read_csv(f'data/test/{symbol}.csv',
                                delimiter=',',
                                usecols=['Open', 'High', 'Low', 'Close', 'Volume'])
        ax = fig.add_subplot(211)
        ax.set_title('Predicted Percentage')
        ax.set_xlabel('Date')
        ax.set_ylabel('Closing Returns (%)')
        ax.plot(test_data['Close'], label='Closing Returns')
        ax.plot(range(seq_len, len(predictions[symbol]) + seq_len),
                predictions[symbol],
                label='Predicted Closing Returns')
        ax.legend(loc='best')
        del test_data

        # Plot predicted stock price
        test_data = pd.read_csv(f'original_data/test/{symbol}.csv',
                                delimiter=',',
                                usecols=['Open', 'High', 'Low', 'Close', 'Volume'])
        ax = fig.add_subplot(212)
        ax.set_title('Predicted Stock Price')
        ax.set_xlabel('Date')
        ax.set_ylabel('Closing Returns')
        ax.plot(test_data['Close'], label='Closing Returns')
        predictions[symbol] = np.add(predictions[symbol], 1.0)
        predicted = np.multiply(test_data['Close'][seq_len + 1:].values.tolist(), predictions[symbol])
        ax.plot(range(seq_len + 1, len(predicted) + seq_len + 1),
                predicted,
                label='Predicted Closing Returns')
        ax.legend(loc='best')

        plt.tight_layout()
        plt.savefig(f'./figures/inference/{symbol}_prediction.png')

        # Print MSE, MAE, and MAPE of the given stock
        original_data = np.array(test_data['Close'][seq_len + 1:].values.tolist())
        predicted = np.array(predicted)
        difference = original_data - predicted
        mse = np.sum(difference ** 2) / len(predicted)
        mae = np.sum(np.abs(difference)) / len(predicted)
        mape = np.sum(np.abs(difference) / original_data) / len(predicted) * 100
        print(f'MSE, MAE, and MAPE of stock {symbol}:   {mse:.4f},   {mae:.4f},   {mape:.4f} %')
