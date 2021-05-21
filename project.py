from argparse import ArgumentParser, ArgumentTypeError, Namespace
import torch.nn as nn
import sys

class DataLoader:
    def __init__(self):
        super(DataLoader, self).__init__()
        # TODO

class Time2Vector:
    def __init__(self):
        super(Time2Vector, self).__init__()
        # TODO

    def forward(self):
        """
        Forward propagation
        :return:
        """
        # TODO


class SingleHeadAttention(nn.Module):
    def __init__(self):
        super(SingleHeadAttention, self).__init__()
        # TODO

    def forward(self):
        """
        Forward propagation
        :return:
        """
        # TODO


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        # TODO

    def forward(self):
        """
        Forward propagation
        :return:
        """
        # TODO


class TransformerEncoder(nn.Module):
    def __init__(self):
        super(TransformerEncoder, self).__init__()
        # TODO

    def forward(self):
        """
        Forward propagation
        :return:
        """
        # TODO


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # TODO

    def forward(self):
        """
        Forward propagation
        :return:
        """
        # TODO


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


def check_verbosity_type(input_value: str) -> int:
    """
    Check whether verbosity is true or false
    :param input_value: input string value
    :return: integer value
    """
    int_value = int(input_value)
    if int_value != 0 and int_value != 1 and int_value != 2:
        raise ArgumentTypeError(f'Verbosity should be 0, 1 or 2.')
    return int_value


def parse_arguments() -> Namespace:
    """
    Parse arguments
    :return: arguments
    """
    parser = ArgumentParser(description='DLP project: Stock Prediction using Transformer')

    parser.add_argument('-v', '--verbosity', default=0, type=check_verbosity_type, help='Verbosity level')

    return parser.parse_args()


def main() -> None:
    """
    Main function
    :return: None
    """
    # Parse arguments
    args = parse_arguments()
    global verbosity
    verbosity = args.verbosity


if __name__ == '__main__':
    verbosity = None
    main()
