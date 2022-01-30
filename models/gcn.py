import argparse
import torch
import torch.nn as nn
from utils.graph_conv import calculate_laplacian


class GCN(nn.Module):
    def __init__(self, adj, input_dim: int, output_dim: int, **kwargs):
        super(GCN, self).__init__()
        self.register_buffer(
            "laplacian", calculate_laplacian(torch.FloatTensor(adj))
        )
        self._num_nodes = adj.shape[0]
        self._input_dim = input_dim
        self._output_dim = output_dim
        self.weights = nn.Parameter(torch.FloatTensor(self._input_dim, self._output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain("tanh"))

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        inputs = inputs.transpose(0, 2).transpose(1, 2)
        inputs = inputs.reshape((self._num_nodes, batch_size * self._input_dim))
        ax = self.laplacian @ inputs
        ax = ax.reshape((self._num_nodes, batch_size, self._input_dim))
        ax = ax.reshape((self._num_nodes * batch_size, self._input_dim))
        outputs = torch.sigmoid(ax @ self.weights)
        outputs = outputs.reshape((self._num_nodes, batch_size, self._output_dim))
        outputs = outputs.transpose(0, 1)
        return outputs

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--hidden_dim", type=int, default=64)
        return parser

    @property
    def hyperparameters(self):
        return {
            "num_nodes": self._num_nodes,
            "input_dim": self._input_dim,
            "output_dim": self._output_dim,
        }
