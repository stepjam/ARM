from typing import List

import torch
import torch.nn as nn

from arm.baselines.sac.sac_agent import SACAgent
from arm.baselines.td3 import launch_utils as td3_launch_utils
from arm.network_utils import Conv2DBlock, DenseBlock, \
    Conv2DUpsampleBlock
from arm.preprocess_agent import PreprocessAgent


def create_replay(*args, **kwargs):
    return td3_launch_utils.create_replay(*args, **kwargs)


def fill_replay(*args, **kwargs):
    return td3_launch_utils.fill_replay(*args, **kwargs)


class Encoder(nn.Module):

    def __init__(self,
                 input_resolutions: List[List[int]],
                 latent_size: int,
                 activation: str):
        super(Encoder, self).__init__()
        self._input_resolutions = input_resolutions
        self._latent_size = latent_size
        self._activation = activation

    def build(self):
        self._rgb_pre = nn.Sequential(
            Conv2DBlock(3, 16, 5, 1, activation=self._activation),
        )
        self._pcd_pre = nn.Sequential(
            Conv2DBlock(3, 16, 5, 1, activation=self._activation),
        )
        self._enc_convs = nn.Sequential(
            Conv2DBlock(16*2, 32, 5, 2, activation=self._activation),  # 128->64
            Conv2DBlock(32, 32, 3, 2, activation=self._activation),    # 64->32
            Conv2DBlock(32, 32, 3, 2, activation=self._activation),    # 32->16
            Conv2DBlock(32, 32, 3, 1, activation=self._activation),    # 16->16
        )
        self._enc_dense = DenseBlock(
            32 * 16 * 16, self._latent_size, norm='layer')

    def forward(self, observations, detach_convs=False):
        x_rgb, x_pcd = self._rgb_pre(observations[0]), self._pcd_pre(observations[1])
        # x = self._pre_fuse(torch.cat([x_rgb, x_pcd], dim=1))
        x = torch.cat([x_rgb, x_pcd], dim=1)
        b, _, h, w = x.shape
        x = self._enc_convs(x)
        if detach_convs:
            x = x.detach()
        x = self._enc_dense(x.view(b, -1))
        return x

    def _tie_weights(self, src, trg):
        assert type(src) == type(trg)
        trg.conv2d.weight = src.conv2d.weight
        trg.conv2d.bias = src.conv2d.bias

    def copy_conv_weights_from(self, source):
        for i in range(4):
            self._tie_weights(src=source._enc_convs[i], trg=self._enc_convs[i])
        self._tie_weights(src=source._rgb_pre[0], trg=self._rgb_pre[0])
        self._tie_weights(src=source._pcd_pre[0], trg=self._pcd_pre[0])


class Decoder(nn.Module):

    def __init__(self,
                 input_resolutions: List[List[int]],
                 latent_size: int,
                 activation: str):
        super(Decoder, self).__init__()
        self._input_resolutions = input_resolutions
        self._latent_size = latent_size
        self._activation = activation

    def build(self):
        self._dec_dense = DenseBlock(
            self._latent_size, 32 * 16 * 16, activation=self._activation)
        self._dec_convs = nn.Sequential(
            Conv2DBlock(32, 32, 3, 1, activation=self._activation),          # 16->16
            Conv2DUpsampleBlock(32, 32, 3, 2, activation=self._activation),  # 16->32
            Conv2DUpsampleBlock(32, 32, 5, 2, activation=self._activation),  # 32->64
            Conv2DUpsampleBlock(32, 32, 5, 2, activation=self._activation),  # 64->128
        )
        self._rgb_pred = Conv2DBlock(32, 3, 7, 1)
        self._pcd_pred = Conv2DBlock(32, 3, 7, 1)

    def forward(self, x):
        x = self._dec_dense(x)
        b = x.shape[0]
        x = self._dec_convs(x.view(b, 32, 16, 16))
        return self._rgb_pred(x), self._pcd_pred(x)


class MLP(nn.Module):

    def __init__(self,
                 output_size: int,
                 latent_size: int,
                 low_dim_state_size: int,
                 activation: str):
        super(MLP, self).__init__()
        self._output_size = output_size
        self._latent_size = latent_size
        self._low_dim_state_size = low_dim_state_size
        self._activation = activation

    def build(self):
        n = 256
        self._low_dim_pre = nn.Sequential(
            DenseBlock(self._low_dim_state_size, n, activation=self._activation),
            DenseBlock(n, self._latent_size, norm='layer'),
        )
        self._mlp = nn.Sequential(
            DenseBlock(self._latent_size * 2, n, activation=self._activation),
            DenseBlock(n, n, activation=self._activation),
            DenseBlock(n, n, activation=self._activation),
            DenseBlock(n, self._output_size)
        )

    def forward(self, x, low_dim_state):
        pre = self._low_dim_pre(low_dim_state)
        fused = torch.cat([x, pre], dim=1)
        return self._mlp(fused)


def create_agent(camera_name: str,
                 activation: str,
                 action_min_max,
                 image_resolution: list,
                 critic_lr,
                 actor_lr,
                 critic_weight_decay,
                 actor_weight_decay,
                 tau,
                 critic_grad_clip,
                 actor_grad_clip,
                 low_dim_state_len,
                 alpha,
                 alpha_auto_tune,
                 alpha_lr,
                 decoder_weight_decay,
                 decoder_grad_clip,
                 decoder_lr,
                 decoder_latent_lambda,
                 encoder_tau):

    latent_size = 50
    decoder_net = Decoder(image_resolution, latent_size, activation)
    encoder_net = Encoder(image_resolution, latent_size, activation)
    critic_net = MLP(1, latent_size, low_dim_state_len + 8, activation)
    actor_net = MLP(8 * 2, latent_size, low_dim_state_len, activation)

    sac_agent = SACAgent(
        critic_network=critic_net,
        actor_network=actor_net,
        decoder_network=decoder_net,
        encoder_network=encoder_net,
        action_min_max=action_min_max,
        camera_name=camera_name,
        critic_lr=critic_lr,
        actor_lr=actor_lr,
        critic_weight_decay=critic_weight_decay,
        actor_weight_decay=actor_weight_decay,
        critic_tau=tau,
        critic_grad_clip=critic_grad_clip,
        actor_grad_clip=actor_grad_clip,
        alpha=alpha,
        alpha_auto_tune=alpha_auto_tune,
        alpha_lr=alpha_lr,
        decoder_weight_decay=decoder_weight_decay,
        decoder_grad_clip=decoder_grad_clip,
        decoder_lr=decoder_lr,
        decoder_latent_lambda=decoder_latent_lambda,
        encoder_tau=encoder_tau
    )

    return PreprocessAgent(pose_agent=sac_agent)
