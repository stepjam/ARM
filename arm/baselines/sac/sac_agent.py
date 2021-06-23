import copy
import logging
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from yarr.agents.agent import Agent, Summary, ActResult, \
    ScalarSummary, HistogramSummary, ImageSummary

from arm import utils
from arm.utils import stack_on_channel

NAME = 'SACAgent'
LOG_STD_MAX = 2
LOG_STD_MIN = -10
REPLAY_ALPHA = 0.7
REPLAY_BETA = 0.5


class QFunction(nn.Module):

    def __init__(self, critic: nn.Module, encoder: nn.Module):
        super(QFunction, self).__init__()
        self._q1 = copy.deepcopy(critic)
        self._q2 = copy.deepcopy(critic)
        self.encoder = copy.deepcopy(encoder)
        self._q1.build()
        self._q2.build()
        self.encoder.build()

    def forward(self, observations, robot_state, action):
        combined = torch.cat([robot_state, action.float()], dim=1)
        latents = self.encoder(observations)
        q1 = self._q1(latents, combined)
        q2 = self._q2(latents, combined)
        return q1, q2


class Actor(nn.Module):

    def __init__(self, actor_network: nn.Module, encoder: nn.Module,
                 action_min_max: torch.tensor):
        super(Actor, self).__init__()
        self._action_min_max = action_min_max
        self._actor_network = copy.deepcopy(actor_network)
        self._actor_network.build()
        self._act_encoder = copy.deepcopy(encoder)
        self._act_encoder.build()

    def _rescale_actions(self, x):
        return (0.5 * (x + 1.) * (
                self._action_min_max[1] - self._action_min_max[0]) +
                self._action_min_max[0])

    def _normalize(self, x):
        return x / x.square().sum(dim=1).sqrt().unsqueeze(-1)

    def _gaussian_logprob(self, noise, log_std):
        residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
        return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)

    def forward(self, observations, robot_state):
        latent = self._act_encoder(observations, detach_convs=True)
        mu_and_logstd = self._actor_network(latent, robot_state)
        mu, log_std = torch.split(mu_and_logstd, 8, dim=1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        std = log_std.exp()
        noise = torch.randn_like(mu)
        pi = mu + noise * std
        log_pi = self._gaussian_logprob(noise, log_std)
        mu = torch.tanh(mu)
        pi = torch.tanh(pi)
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)

        pi = self._rescale_actions(pi)
        mu = self._rescale_actions(mu)

        pi = torch.cat(
            [pi[:, :3], self._normalize(pi[:, 3:7]), pi[:, 7:]], dim=-1)
        mu = torch.cat(
            [mu[:, :3], self._normalize(mu[:, 3:7]), mu[:, 7:]], dim=-1)
        return mu, pi, log_pi, log_std

    def get_params(self):
        return list(self._act_encoder._enc_dense.parameters()
                    ) + list(self._actor_network.parameters())


class SACAgent(Agent):

    def __init__(self,
                 critic_network: nn.Module,
                 actor_network: nn.Module,
                 decoder_network: nn.Module,
                 encoder_network: nn.Module,
                 action_min_max: tuple,
                 camera_name: str,
                 critic_lr: float = 0.01,
                 actor_lr: float = 0.01,
                 critic_weight_decay: float = 1e-5,
                 actor_weight_decay: float = 1e-5,
                 critic_tau: float = 0.005,
                 critic_grad_clip: float = 20.0,
                 actor_grad_clip: float = 20.0,
                 gamma: float = 0.99,
                 nstep: int = 1,
                 alpha: float = 0.2,
                 alpha_auto_tune: bool = True,
                 alpha_lr: float = 0.0001,
                 decoder_weight_decay=1e-6,
                 decoder_grad_clip=5,
                 decoder_lr=0.001,
                 decoder_latent_lambda=1e-6,
                 encoder_tau=0.05):
        self._decoder_network = decoder_network
        self._encoder_network = encoder_network
        self._critic_tau = critic_tau
        self._critic_grad_clip = critic_grad_clip
        self._actor_grad_clip = actor_grad_clip
        self._camera_name = camera_name
        self._gamma = gamma
        self._nstep = nstep
        self._critic_network = critic_network
        self._actor_network = actor_network
        self._action_min_max = action_min_max
        self._critic_lr = critic_lr
        self._actor_lr = actor_lr
        self._critic_weight_decay = critic_weight_decay
        self._actor_weight_decay = actor_weight_decay
        self._alpha = alpha
        self._alpha_auto_tune = alpha_auto_tune
        self._alpha_lr = alpha_lr
        self._target_entropy = -len(action_min_max[0])

        self._decoder_weight_decay = decoder_weight_decay
        self._decoder_grad_clip = decoder_grad_clip
        self._decoder_lr = decoder_lr
        self._decoder_latent_lambda = decoder_latent_lambda
        self._encoder_tau = encoder_tau

    def build(self, training: bool, device: torch.device = None):
        if device is None:
            device = torch.device('cpu')

        self._action_min_max = torch.tensor(self._action_min_max).float().to(device)

        self._actor = Actor(self._actor_network, self._encoder_network, self._action_min_max).to(
            device).train(training)

        if training:
            self._q = QFunction(self._critic_network, self._encoder_network).to(device).train(training)
            self._actor._act_encoder.copy_conv_weights_from(self._q.encoder)
            self._q_target = QFunction(self._critic_network, self._encoder_network).to(device).train(False)
            utils.soft_updates(self._q, self._q_target, 1.0)

            self._decoder = copy.deepcopy(self._decoder_network)
            self._decoder.build()
            self._decoder = self._decoder.to(device).train(training)

            # Freeze target critic.
            for p in self._q_target.parameters():
                p.requires_grad = False

            self._critic_optimizer = torch.optim.Adam(
                list(self._q.parameters()), lr=self._critic_lr,
                weight_decay=self._critic_weight_decay)

            self._encoder_optimizer = torch.optim.Adam(
                self._q.encoder.parameters(), lr=self._decoder_lr)

            self._decoder_optimizer = torch.optim.Adam(
                self._decoder.parameters(), lr=self._decoder_lr,
                weight_decay=self._decoder_weight_decay)

            self._actor_optimizer = torch.optim.Adam(
                self._actor.get_params(), lr=self._actor_lr,
                weight_decay=self._actor_weight_decay)

            self._log_alpha = 0
            if self._alpha_auto_tune:
                self._log_alpha = torch.tensor(
                    (np.log(self._alpha)), dtype=torch.float,
                    requires_grad=True, device=device)
                if training:
                    self._alpha_optimizer = torch.optim.Adam(
                        [self._log_alpha], lr=self._alpha_lr)
            else:
                self._alpha = torch.tensor(
                    self._alpha, dtype=torch.float,
                    requires_grad=False, device=device)

            logging.info('# TD3 Critic Params: %d' % sum(
                p.numel() for p in self._q.parameters() if p.requires_grad))
            logging.info('# TD3 Actor Params: %d' % sum(
                p.numel() for p in self._actor.parameters() if p.requires_grad))
            logging.info('# TD3 Decoder Params: %d' % sum(
                p.numel() for p in self._decoder.parameters() if p.requires_grad))
        else:
            for p in self._actor.parameters():
                p.requires_grad = False

        self._device = device

    def _preprocess_inputs(self, replay_sample):
        observations = [
            stack_on_channel(replay_sample['%s_rgb' % self._camera_name]),
            stack_on_channel(
                replay_sample['%s_point_cloud' % self._camera_name])
        ]
        tp1_observations = [
            stack_on_channel(replay_sample['%s_rgb_tp1' % self._camera_name]),
            stack_on_channel(
                replay_sample['%s_point_cloud_tp1' % self._camera_name])
        ]
        return observations, tp1_observations

    def _clip_action(self, a):
        return torch.min(torch.max(a, self._action_min_max[0].unsqueeze(0)),
                         self._action_min_max[1])

    def _update_critic(self, replay_sample: dict, reward) -> None:
        action = replay_sample['action']

        robot_state = stack_on_channel(replay_sample['low_dim_state'][:, -1:])
        robot_state_tp1 = stack_on_channel(
            replay_sample['low_dim_state_tp1'][:, -1:])

        terminal = replay_sample['terminal'].float()

        observations, tp1_observations = self._preprocess_inputs(replay_sample)

        q1, q2 = self._q(observations, robot_state, action)

        with torch.no_grad():

            _, pi_tp1, logp_pi_tp1, _ = self._actor(
                tp1_observations, robot_state_tp1)

            q1_pi_tp1_targ, q2_pi_tp1_targ = self._q_target(
                tp1_observations, robot_state_tp1, pi_tp1)

            next_value = torch.min(q1_pi_tp1_targ, q2_pi_tp1_targ)
            next_value = (next_value - self.alpha.detach() * logp_pi_tp1)
            q_backup = (reward.unsqueeze(-1) + (
                        self._gamma ** self._nstep) * (
                            1. - terminal.unsqueeze(-1)) * next_value)

        loss_weights = utils.loss_weights(replay_sample, REPLAY_BETA)
        q1_delta = F.smooth_l1_loss(q1, q_backup, reduction='none')
        q2_delta = F.smooth_l1_loss(q2, q_backup, reduction='none')

        q1_delta, q2_delta = q1_delta.mean(1), q2_delta.mean(1)
        q1_bellman_loss = (q1_delta * loss_weights).mean()
        q2_bellman_loss = (q2_delta * loss_weights).mean()
        critic_loss = (q1_bellman_loss + q2_bellman_loss)

        self._critic_summaries = {
            'q1_bellman_loss': q1_bellman_loss,
            'q2_bellman_loss': q2_bellman_loss,
            'q1_mean': q1.mean().item(),
            'q2_mean': q2.mean().item(),
            'alpha': self.alpha,
        }
        new_pri = torch.sqrt((q1_delta + q2_delta) / 2. + 1e-10)
        self._new_priority = (new_pri / torch.max(new_pri)).detach()
        self._grad_step(critic_loss, self._critic_optimizer,
                        list(self._q.parameters()), self._critic_grad_clip)

    def _update_actor(self, replay_sample: dict) -> None:

        robot_state = stack_on_channel(replay_sample['low_dim_state'][:, -1:])
        observations = [
            stack_on_channel(replay_sample['%s_rgb' % self._camera_name]),
            stack_on_channel(
                replay_sample['%s_point_cloud' % self._camera_name])
        ]

        mu, pi, self._logp_pi, log_scale_diag = self._actor(observations, robot_state)
        q1_pi, q2_pi = self._q(observations, robot_state, pi)
        pi_loss = self.alpha.detach() * self._logp_pi - torch.min(q1_pi, q2_pi)

        loss_weights = utils.loss_weights(replay_sample, REPLAY_BETA)
        pi_loss = (pi_loss * loss_weights).mean()

        self._actor_summaries = {
            'pi/loss': pi_loss,
            'pi/q1_pi_mean': q1_pi.mean(),
            'pi/pi': pi.mean(),
            'pi/mu': mu.mean(),
            'pi/log_pi': self._logp_pi.mean(),
            'pi/log_scale_diag': log_scale_diag.mean()
        }
        self._grad_step(pi_loss, self._actor_optimizer,
                        self._actor.get_params(), self._actor_grad_clip)

    def _get_recon_losses(self, observations):
        latent = self._q.encoder(observations)
        rgb_recon, pcd_recon = self._decoder(latent)
        latent_loss = (0.5 * latent.pow(2).sum(
            1)).mean() * self._decoder_latent_lambda
        rec_loss = F.mse_loss(
            rgb_recon, observations[0], reduction='none') + F.mse_loss(
            pcd_recon, observations[1], reduction='none')
        rec_loss = rec_loss.mean(-1).mean(-1).mean(-1)
        return rec_loss, latent_loss, latent, rgb_recon, pcd_recon

    def _update_decoder(self, replay_sample: dict) -> None:
        observations, tp1_observations = self._preprocess_inputs(replay_sample)
        rec_loss, latent_loss, latent, rgb_recon, pcd_recon = self._get_recon_losses(observations)
        rec_loss_tp1, latent_loss_tp1, _, _, _ = self._get_recon_losses(tp1_observations)
        rec_loss += rec_loss_tp1
        latent_loss += latent_loss_tp1

        total_loss = latent_loss + rec_loss.mean()

        self._new_priority += (rec_loss / torch.max(rec_loss)).detach()

        self._decoder_summaries = {
            'decoder/loss': total_loss,
            'decoder/latent_loss': latent_loss,
            'decoder/recon_loss': rec_loss.mean(),
            'decoder/latent': latent.mean(),
            'decoder/latent_min': latent.min(),
            'decoder/latent_max': latent.max(),
        }
        self._decoder_image_summaries = {
            'decoder/rgb_recon': torch.clamp((rgb_recon[:1] + 1.0) / 2.0, 0, 1),
            'decoder/pcd_recon': pcd_recon[:1],
        }
        self._encoder_optimizer.zero_grad()
        self._decoder_optimizer.zero_grad()
        total_loss.backward()
        self._encoder_optimizer.step()
        self._decoder_optimizer.step()

    def _update_alpha(self):
        alpha_loss = -(self.alpha * (
                self._logp_pi + self._target_entropy).detach()).mean()
        self._grad_step(alpha_loss, self._alpha_optimizer)

    def _grad_step(self, loss, opt, model_params=None, clip=None):
        opt.zero_grad()
        loss.backward()
        if clip is not None and model_params is not None:
            nn.utils.clip_grad_value_(model_params, clip)
        opt.step()

    @property
    def alpha(self):
        return self._log_alpha.exp() if self._alpha_auto_tune else self._alpha

    def update(self, step: int, replay_sample: dict) -> dict:
        reward = replay_sample['reward']
        self._update_critic(replay_sample, reward)

        # Freeze critic so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self._q.parameters():
            p.requires_grad = False

        if step % 2 == 0:
            self._update_actor(replay_sample)
            if self._alpha_auto_tune:
                self._update_alpha()

        # UnFreeze critic.
        for p in self._q.parameters():
            p.requires_grad = True

        self._update_decoder(replay_sample)

        if step % 2 == 0:
            utils.soft_updates(self._q._q1, self._q_target._q1, self._critic_tau)
            utils.soft_updates(self._q._q2, self._q_target._q2, self._critic_tau)
            utils.soft_updates(self._q.encoder, self._q_target.encoder, self._encoder_tau)
        return {
            'priority': self._new_priority ** REPLAY_ALPHA
        }

    def act(self, step: int, observation: dict,
            deterministic=False) -> ActResult:
        with torch.no_grad():
            observations = [
                stack_on_channel(observation['%s_rgb' % self._camera_name]),
                stack_on_channel(
                    observation['%s_point_cloud' % self._camera_name])
            ]
            robot_state = stack_on_channel(observation['low_dim_state'][:, -1:])
            mu, pi, _, _ = self._actor(observations, robot_state)
            return ActResult((mu if deterministic else pi)[0])

    def update_summaries(self) -> List[Summary]:

        summaries = []
        for n, v in list(self._critic_summaries.items()) + list(
                self._actor_summaries.items()) + list(
                self._decoder_summaries.items()):
            summaries.append(ScalarSummary('%s/%s' % (NAME, n), v))

        for n, v in self._decoder_image_summaries.items():
            summaries.append(ImageSummary('%s/%s' % (NAME, n), v))

        for tag, param in list(self._q.named_parameters()) + list(
                self._actor.named_parameters()) + list(
                self._decoder.named_parameters()):
            summaries.append(
                HistogramSummary('%s/gradient/%s' % (NAME, tag), param.grad))
            summaries.append(
                HistogramSummary('%s/weight/%s' % (NAME, tag), param.data))

        return summaries

    def act_summaries(self) -> List[Summary]:
        return []

    def load_weights(self, savedir: str):
        self._actor.load_state_dict(
            torch.load(os.path.join(savedir, 'pose_actor.pt'),
                       map_location=torch.device('cpu')))

    def save_weights(self, savedir: str):
        torch.save(self._actor.state_dict(),
                   os.path.join(savedir, 'pose_actor.pt'))
