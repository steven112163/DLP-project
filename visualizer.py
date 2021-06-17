from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import pandas as pd


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
        ax.set_title("Training Data")
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
        ax.set_title("Testing Data")
        ax.set_xlabel('Date')
        ax.set_ylabel('Closing Returns')
        ax.plot(test_data['Close'], label='Closing Returns')
        ax.plot(range(seq_len, len(test_predictions[symbol]) + seq_len),
                test_predictions[symbol],
                linewidth=3,
                label='Predicted Closing Returns')
        ax.legend(loc='best')
        del test_data

        plt.tight_layout()
        plt.savefig(f'./figures/{symbol}_prediction.png')
