from typing import Tuple
import torch.nn as nn
import torch


class Time2Vector(nn.Module):
    def __init__(self, seq_len: int):
        super(Time2Vector, self).__init__()

        self.weights_non_periodic = nn.Parameter(data=torch.randn(seq_len))
        self.bias_non_periodic = nn.Parameter(data=torch.randn(seq_len))
        self.weights_periodic = nn.Parameter(data=torch.randn(seq_len))
        self.bias_periodic = nn.Parameter(data=torch.randn(seq_len))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation
        :param inputs: stock price [batch_size, seq_len, 5]
        :return: time feature [batch_size, seq_len, 2]
        """
        x = torch.mean(inputs[:, :, 0:4], dim=-1)
        time_non_periodic = x * self.weights_non_periodic + self.bias_non_periodic
        time_non_periodic = torch.unsqueeze(time_non_periodic, dim=-1)

        time_periodic = torch.sin(x * self.weights_periodic + self.bias_periodic)
        time_periodic = torch.unsqueeze(time_periodic, dim=-1)

        return torch.cat([time_non_periodic, time_periodic], dim=-1)


class SingleHeadAttention(nn.Module):
    def __init__(self, attn_dim: int):
        super(SingleHeadAttention, self).__init__()
        # TODO
        self.attn_dim = attn_dim

        self.query = nn.Linear(in_features=7, out_features=self.attn_dim)
        self.key = nn.Linear(in_features=7, out_features=self.attn_dim)
        self.value = nn.Linear(in_features=7, out_features=self.attn_dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Forward propagation
        :param inputs: (query, key, value), each dimension is [batch_size, seq_len, 7]
        :return: attention weight [batch_size, seq_len, attn_dim]
        """
        q = self.query(inputs[0])
        k = self.key(inputs[1])
        v = self.value(inputs[2])

        attn_weights = torch.matmul(q, torch.transpose(k, 1, 2))
        attn_weights = torch.div(attn_weights, self.attn_dim ** 0.5)
        attn_weights = self.softmax(attn_weights)
        return torch.matmul(attn_weights, v)


class MultiHeadAttention(nn.Module):
    def __init__(self, attn_dim: int, num_heads: int):
        super(MultiHeadAttention, self).__init__()

        self.attn_heads = list()
        for _ in range(num_heads):
            self.attn_heads.append(SingleHeadAttention(attn_dim=attn_dim))

        self.linear = nn.Linear(in_features=num_heads * attn_dim, out_features=7)

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Forward propagation
        :param inputs: (query, key, value), each dimension is [batch_size, seq_len, 7]
        :return: attention weight [batch_size, seq_len, 7]
        """
        attn = [head.forward(inputs=inputs) for head in self.attn_heads]
        concat_attn = torch.cat(attn, dim=-1)
        return self.linear(concat_attn)


class TransformerEncoder(nn.Module):
    def __init__(self,
                 batch_size: int,
                 seq_len: int,
                 attn_dim: int,
                 num_heads: int,
                 dropout_rate: float,
                 hidden_size: int):
        super(TransformerEncoder, self).__init__()

        self.first = nn.Sequential(
            MultiHeadAttention(attn_dim=attn_dim,
                               num_heads=num_heads),
            nn.Dropout(p=dropout_rate)
        )

        self.second = nn.Sequential(
            nn.LayerNorm(normalized_shape=[batch_size, seq_len, 7]),
            nn.Linear(in_features=7,
                      out_features=hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hidden_size,
                      out_features=7),
            nn.Dropout(p=dropout_rate)
        )

        self.layer_normalization = nn.LayerNorm(normalized_shape=[batch_size, seq_len, 7])

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Forward propagation
        :param inputs: (query, key, value), each dimension is [batch_size, seq_len, 7]
        :return: embedding [batch_size, seq_len, 7]
        """
        query = inputs[0]
        partial_results = self.first(inputs)
        partial_results = self.second(query + partial_results)
        return self.layer_normalization(query + partial_results)


class Network(nn.Module):
    def __init__(self,
                 batch_size: int,
                 seq_len: int,
                 num_encoder: int,
                 attn_dim: int,
                 num_heads: int,
                 dropout_rate: float,
                 hidden_size: int):
        super(Network, self).__init__()

        self.batch_size = batch_size
        self.seq_len = seq_len

        self.time_embedding = Time2Vector(seq_len=seq_len)

        self.encoder = [
            TransformerEncoder(batch_size=batch_size,
                               seq_len=seq_len,
                               attn_dim=attn_dim,
                               num_heads=num_heads,
                               dropout_rate=dropout_rate,
                               hidden_size=hidden_size) for _ in range(num_encoder)
        ]
        self.encoder = nn.Sequential(*self.encoder)

        self.average = nn.AvgPool1d(kernel_size=7)

        self.net = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=seq_len,
                      out_features=64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=64,
                      out_features=1)
        )

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Forward propagation
        :param inputs: stock price [batch_size, seq_len, 5]
        :return: prediction
        """
        time_embedding = self.time_embedding(inputs)
        embedded_inputs = torch.cat([inputs, time_embedding], dim=-1)
        embedded_inputs = self.encoder((embedded_inputs for _ in range(3)))
        embedded_inputs = self.average(embedded_inputs)
        embedded_inputs = embedded_inputs.view(self.batch_size, self.seq_len)
        return self.net(embedded_inputs)