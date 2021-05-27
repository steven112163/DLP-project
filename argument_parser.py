from argparse import ArgumentParser, ArgumentTypeError, Namespace


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

    parser.add_argument('-e', '--epochs', default=100, type=int, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('-s', '--seq_len', default=128, type=int, help='Sequence length (consecutive days)')
    parser.add_argument('-ne', '--num_encoders', default=3, type=int,
                        help='Number of transformer encoder in the network')
    parser.add_argument('-a', '--attn_dim', default=96, type=int, help='Dimension of single attention output')
    parser.add_argument('-nh', '--num_heads', default=12, type=int, help='Number of heads for multi-attention')
    parser.add_argument('-d', '--dropout_rate', default=0.1, type=float, help='Dropout rate')
    parser.add_argument('-hs', '--hidden_size', default=256, type=int,
                        help='Hidden size between the linear layers in the network')
    parser.add_argument('-v', '--verbosity', default=0, type=check_verbosity_type, help='Verbosity level')

    return parser.parse_args()
