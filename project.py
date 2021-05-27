from argument_parser import parse_arguments
from model import Network
from data_loader import DataLoader
import torch.nn as nn
import torch
import sys


def train() -> None:
    """
    Training
    :return: None
    """
    # TODO


def test() -> None:
    """
    Testing
    :return: None
    """
    # TODO


def info_log(log: str) -> None:
    """
    Print information log
    :param log: log to be displayed
    :return: None
    """
    global verbosity
    if verbosity:
        print(f'[\033[96mINFO\033[00m] {log}')
        sys.stdout.flush()


def debug_log(log: str) -> None:
    """
    Print debug log
    :param log: log to be displayed
    :return: None
    """
    global verbosity
    if verbosity > 1:
        print(f'[\033[93mDEBUG\033[00m] {log}')
        sys.stdout.flush()


def main() -> None:
    """
    Main function
    :return: None
    """
    # Parse arguments
    args = parse_arguments()
    info_log(f'Number of epochs: {args.epochs}')
    info_log(f'Batch size: {args.batch_size}')
    info_log(f'Sequence length (consecutive days): {args.seq_len}')
    global verbosity
    verbosity = args.verbosity


if __name__ == '__main__':
    verbosity = None
    main()
