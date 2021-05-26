from argparse import ArgumentParser, ArgumentTypeError, Namespace
import torch.nn as nn
import sys

class DataLoader:
    def __init__(self):
        super(DataLoader, self).__init__()
        # TODO

class Time2Vector:
    def __init__(self, seq_len, **kwargs):
        super(Time2Vector, self).__init__()
        # TODO
        self.seq_len = seq_len
        self.weights_nonperiodic = nn.Parameter(torch.zeros(seq_len))
        self.bias_nonperiodic = nn.Parameter(torch.zeros(seq_len))
        self.weights_periodic = nn.Parameter(torch.zeros(seq_len))
        self.bias_periodic = nn.Parameter(torch.zeros(seq_len))

    def forward(self, x):
        """
        Forward propagation
        :return:
        """
        # TODO
        x = torch.mean(x[:,:,0:4], dim=-1)
        time_nonperiodic = x * self.weights_nonperiodic + self.bias_nonperiodic
        time_nonperiodic = np.expand_dims(time_nonperiodic, axis = -1)

        time_periodic = torch.sin(x * self.weights_periodic + self.bias_periodic)
        time_periodic = np.expand_dims(time_periodic, axis = -1)

        return np.concatenate([time_nonperiodic, time_periodic], axis= -1)
        

class SingleHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, attribute=7):
        super(SingleHeadAttention, self).__init__()
        # TODO
        self.d_k = d_k
        self.d_v = d_v
        self.query = nn.Linear(attribute, self.d_k)
        self.key = nn.Linear(attribute, self.d_k)
        self.value = nn.Linear(attribute, self.d_v)

    def forward(self, x): #x=[q,k,v]
        """
        Forward propagation
        :return:
        """
        # TODO
        q = self.query(x[0])
        k = self.key(x[1])
        v = self.value(x[2])

        attn_weights = np.matmul(q, k.T)
        for w in attn_weights:
            w /= np.sqrt(self.d_k)
        attn_weights = nn.Softmax(attn_weights, dim=-1)
        out = np.matmul(attn_weights, v) 
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, n_heads, attribute=288):
        super(MultiHeadAttention, self).__init__()
        # TODO
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.attn_heads = list()
        for n in range(self.n_heads):
            self.attn_heads.append(SingleHeadAttention(self.d_k, self.d_v))
        self.linear = nn.Linear(attribute, 7)

    def forward(self, x):
        """
        Forward propagation
        :return:
        """
        # TODO
        attn = [self.attn_heads[i](x) for i in range(self.attn_heads)]
        concat_attn = np.concatenate(attn, axis=-1)
        out = self.linear(concat_attn)


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
