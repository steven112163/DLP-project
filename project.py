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
    # Get training device
    training_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Parse arguments
    args = parse_arguments()
    global verbosity
    verbosity = args.verbosity
    info_log(f'Number of epochs: {args.epochs}')
    info_log(f'Batch size: {args.batch_size}')
    info_log(f'Sequence length (consecutive days): {args.seq_len}')
    info_log(f'Number of transformer encoder: {args.num_encoders}')
    info_log(f'Dimension of single attention output: {args.attn_dim}')
    info_log(f'Number of heads for multi-attention: {args.num_heads}')
    info_log(f'Dropout rate: {args.dropout_rate}')
    info_log(f'Hidden size between the linear layers in the network: {args.hidden_size}')
    info_log(f'Training device: {training_device}')

    # Setup model
    info_log('Setup model ...')
    model = Network(batch_size=args.batch_size,
                    seq_len=args.seq_len,
                    num_encoders=args.num_encoders,
                    attn_dim=args.attn_dim,
                    num_heads=args.num_heads,
                    dropout_rate=args.dropout_rate,
                    hidden_size=args.hidden_size).to(training_device)


if __name__ == '__main__':
    verbosity = None
    main()
