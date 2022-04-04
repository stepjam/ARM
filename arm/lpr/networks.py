from typing import List
import torch
from torch import nn
from arm.network_utils import DenseBlock


class DenseNet(nn.Module):

    def __init__(self,
                 input_size: int,
                 fc_layers: List[int],
                 activation: str = 'relu'):
        super(DenseNet, self).__init__()
        self._input_size = input_size
        self._fc_layers = fc_layers
        self._activation = activation

    def build(self):
        dense_layers = []
        channels = self._input_size
        for n in self._fc_layers[:-1]:
            dense_layers.append(
                DenseBlock(channels, n, activation=self._activation))
            channels = n
        dense_layers.append(DenseBlock(channels, self._fc_layers[-1]))
        self._fcs = nn.Sequential(*dense_layers)

    def forward(self, x):
        return self._fcs(x)


class PointFeatCNN(nn.Module):
    def __init__(self, input_dims):
        super(PointFeatCNN, self).__init__()
        self.net = torch.nn.Sequential(
            nn.Conv1d(input_dims, 64, kernel_size=1),
            nn.PReLU(),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.PReLU(),
            nn.Conv1d(128, 1024, kernel_size=1),
            nn.AdaptiveMaxPool1d(output_size=1)
        )

    def forward(self, x):
        x = self.net(x)
        return x[:, :, 0]


class PointNet(nn.Module):
    def __init__(self, input_dims, extra_dims, output_dims=1):
        super(PointNet, self).__init__()
        self._input_dims = input_dims
        self._extra_dims = extra_dims
        self._output_dims = output_dims

    def build(self):
        self.feat_net = PointFeatCNN(self._input_dims)
        head_ins = 1024
        if self._extra_dims > 0:
            self.extra_pre = nn.Sequential(
                nn.Linear(self._extra_dims, 1024),
                nn.PReLU(),
            )
            head_ins *= 2
        self.head = nn.Sequential(
            nn.Linear(head_ins, 256),
            nn.PReLU(),
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Linear(128, self._output_dims)
        )

    def forward(self, x):
        # x will be trajectory_points * 7
        bs = x.shape[0]
        feats_x = self.feat_net(x[:, :-self._extra_dims].view(bs, 7, -1))  # TODO
        if self._extra_dims > 0:
            feats_extra = self.extra_pre(x[:, -self._extra_dims:])
            feats_x = torch.cat([feats_x, feats_extra], 1)
        out = self.head(feats_x)
        return out
