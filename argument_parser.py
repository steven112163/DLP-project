from argparse import ArgumentParser, Namespace


def parse_arguments() -> Namespace:
    """
    Parse arguments
    :return: Arguments
    """
    parser = ArgumentParser(description='DLP project: Stock Prediction using Transformer')

    parser.add_argument('-e', '--epochs', default=100, type=int, help='Number of epochs')
    parser.add_argument('-w', '--warmup', default=10, type=int, help='Number of epochs for warmup')
    parser.add_argument('-l', '--learning_rate', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('-b', '--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('-s', '--seq_len', default=128, type=int, help='Sequence length (consecutive days)')
    parser.add_argument('-ne', '--num_encoders', default=3, type=int,
                        help='Number of transformer encoder in the network')
    parser.add_argument('-a', '--attn_dim', default=96, type=int, help='Dimension of single attention output')
    parser.add_argument('-nh', '--num_heads', default=12, type=int, help='Number of heads for multi-attention')
    parser.add_argument('-d', '--dropout_rate', default=0.1, type=float, help='Dropout rate')
    parser.add_argument('-hs', '--hidden_size', default=256, type=int,
                        help='Hidden size between the linear layers in the network')
    parser.add_argument('-i', '--inference_only', action='store_true', help='Inference only or not')
    parser.add_argument('-r', '--root_dir', default='archive', type=str,
                        help='Directory containing the downloaded data')
    parser.add_argument('-v', '--verbosity', default=0, type=int, choices=[0, 1, 2], help='Verbosity level')

    return parser.parse_args()
