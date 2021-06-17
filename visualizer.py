from typing import List, Tuple
import matplotlib.pyplot as plt


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
