import copy
import logging
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from yarr.agents.agent import Agent, Summary, ActResult, \
    ScalarSummary, HistogramSummary

from arm import utils
from arm.utils import stack_on_channel

NAME = 'TD3Agent'
REPLAY_ALPHA = 0.7
REPLAY_BETA = 0.4


class QFunction(nn.Module):

    def __init__(self, critic: nn.Module):
        super(QFunction, self).__init__()
        self._q1 = copy.deepcopy(critic)
        self._q2 = copy.deepcopy(critic)
        self._q1.build()
        self._q2.build()

    def forward(self, observations, robot_state, action, q1_only=False):
        combined = torch.cat([robot_state, action.float()], dim=1)
        q1 = self._q1(observations, combined)
        q2 = None if q1_only else self._q2(observations, combined)
        return q1, q2


class Actor(nn.Module):

    def __init__(self, actor_network: nn.Module, action_min_max: torch.tensor):
        super(Actor, self).__init__()
        self._action_min_max = action_min_max
        self._actor_network = copy.deepcopy(actor_network)
        self._actor_network.build()

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
        pi = self._actor_network(observations, robot_state)
        pi = torch.tanh(pi)
        pi = self._rescale_actions(pi)
        pi = torch.cat([pi[:, :3], self._normalize(pi[:, 3:7]), pi[:, 7:]], dim=-1)
        return pi


class TD3Agent(Agent):

    def __init__(self,
                 critic_network: nn.Module,
                 actor_network: nn.Module,
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
                 explore_noise: float = 0.1,
                 smoothing_noise: float = 0.2):
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
        self._explore_noise = explore_noise
        self._smoothing_noise = smoothing_noise
        self._smoothing_noise_clip = smoothing_noise * 2

    def build(self, training: bool, device: torch.device = None):
        if device is None:
            device = torch.device('cpu')

        self._action_min_max = torch.tensor(self._action_min_max).to(device)
        self._actor = Actor(self._actor_network, self._action_min_max).to(
            device).train(training)

        if training:
            self._q = QFunction(self._critic_network).to(device).train(training)
            self._q_target = QFunction(self._critic_network).to(device).train(False)
            utils.soft_updates(self._q, self._q_target, 1.0)

            # Freeze target critic.
            for p in self._q_target.parameters():
                p.requires_grad = False

            self._critic_optimizer = torch.optim.Adam(
                self._q.parameters(), lr=self._critic_lr,
                weight_decay=self._critic_weight_decay)
            self._actor_optimizer = torch.optim.Adam(
                self._actor.parameters(), lr=self._actor_lr,
                weight_decay=self._actor_weight_decay)

            logging.info('# TD3 Critic Params: %d' % sum(
                p.numel() for p in self._q.parameters() if p.requires_grad))
            logging.info('# TD3 Actor Params: %d' % sum(
                p.numel() for p in self._actor.parameters() if p.requires_grad))
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

        # Don't want timeouts to be classed as terminals
        terminal = replay_sample['terminal'].float()

        observations, tp1_observations = self._preprocess_inputs(replay_sample)
        q1, q2 = self._q(observations, robot_state, action)

        with torch.no_grad():

            pi_tp1 = self._actor(
                tp1_observations, robot_state_tp1)

            # Target policy smoothing
            epsilon = torch.randn_like(pi_tp1) * self._smoothing_noise
            epsilon = torch.clamp(epsilon, -self._smoothing_noise_clip,
                                  self._smoothing_noise_clip)
            a2 = self._clip_action(pi_tp1 + epsilon)

            q1_pi_tp1_targ, q2_pi_tp1_targ = self._q_target(
                tp1_observations, robot_state_tp1, a2)

            next_value = torch.min(q1_pi_tp1_targ, q2_pi_tp1_targ)
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
        }
        new_pri = torch.sqrt((q1_delta + q2_delta) / 2. + 1e-10)
        self._new_priority = (new_pri / torch.max(new_pri)).detach()
        self._grad_step(critic_loss, self._critic_optimizer,
                        self._q.parameters(), self._critic_grad_clip)

    def _update_actor(self, replay_sample: dict) -> None:

        robot_state = stack_on_channel(replay_sample['low_dim_state'][:, -1:])
        observations = [
            stack_on_channel(replay_sample['%s_rgb' % self._camera_name]),
            stack_on_channel(
                replay_sample['%s_point_cloud' % self._camera_name])
        ]

        pi = self._actor(observations, robot_state)
        q1_pi, _ = self._q(observations, robot_state, pi, q1_only=True)
        pi_loss = -q1_pi

        loss_weights = utils.loss_weights(replay_sample, REPLAY_BETA)
        pi_loss = (pi_loss * loss_weights).mean()

        self._actor_summaries = {
            'pi/loss': pi_loss,
            'pi/q1_pi_mean': q1_pi.mean(),
            'pi/pi': pi.mean(),
        }
        self._grad_step(pi_loss, self._actor_optimizer,
                        self._actor.parameters(), self._actor_grad_clip)


    def _grad_step(self, loss, opt, model_params=None, clip=None):
        opt.zero_grad()
        loss.backward()
        if clip is not None and model_params is not None:
            nn.utils.clip_grad_value_(model_params, clip)
        opt.step()

    def update(self, step: int, replay_sample: dict) -> dict:

        reward = replay_sample['reward']
        self._update_critic(replay_sample, reward)

        # Freeze critic so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self._q.parameters():
            p.requires_grad = False

        if step % 2 == 0:
            self._update_actor(replay_sample)

        # UnFreeze critic.
        for p in self._q.parameters():
            p.requires_grad = True

        utils.soft_updates(self._q, self._q_target, self._critic_tau)
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
            pi = self._actor(observations, robot_state)
            if not deterministic:
                pi += self._explore_noise * torch.randn_like(pi)
                pi = self._clip_action(pi)
                pi = torch.cat(
                    [pi[:, :3], self._actor._normalize(pi[:, 3:7]), pi[:, 7:]], dim=-1)
            return ActResult(pi[0])

    def update_summaries(self) -> List[Summary]:

        summaries = []
        for n, v in list(self._critic_summaries.items()) + list(
                self._actor_summaries.items()):
            summaries.append(ScalarSummary('%s/%s' % (NAME, n), v))

        for tag, param in list(self._q.named_parameters()) + list(
                self._actor.named_parameters()):
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
