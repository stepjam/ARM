import copy
import logging
import os
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from yarr.agents.agent import Agent, Summary, ActResult, \
    ScalarSummary, HistogramSummary

from arm import utils
from arm.lpr.const import UNKNOWN_COLLISION, CFG_MIN_ACTIONS, CFG_MAX_ACTIONS
from arm.utils import stack_on_channel

NAME = 'TrajectoryAgent'
REPLAY_ALPHA = 1.0
REPLAY_BETA = 1.0


class QFunction(nn.Module):

    def __init__(self, q: nn.Module):
        super(QFunction, self).__init__()
        self._q1 = copy.deepcopy(q)
        self._q2 = copy.deepcopy(q)
        self._q1.build()
        self._q2.build()

    def forward(self, state_and_action):
        time = state_and_action[:, -1:]
        collision_onehot = state_and_action[:, -4:-1]
        tmp = state_and_action[:, :-4].view(state_and_action.shape[0], -1, 7)
        start = tmp[:, :1, :3]
        state_and_action = torch.cat(
            [(tmp[:, :, :3] - start), tmp[:, :, 3:]], 2).reshape(
            state_and_action.shape[0], -1)
        state_and_action = torch.cat(
            [state_and_action, collision_onehot, time], 1)
        q1 = self._q1(state_and_action)
        q2 = self._q2(state_and_action)
        return q1, q2


class Actor(nn.Module):

    def __init__(self, actor_network: nn.Module, action_min_max):
        super(Actor, self).__init__()
        self._action_min_max = action_min_max
        self._actor_network = copy.deepcopy(actor_network)
        self._actor_network.build()

    def _rescale_actions(self, x):
        return (0.5 * (x + 1.) * (
                self._action_min_max[1] - self._action_min_max[0]) +
                self._action_min_max[0])

    def _normalize(self, x):
        return x / x.square().sum(dim=-1).sqrt().unsqueeze(-1)

    def forward(self, state):
        pi = self._actor_network(state)
        return pi


class TrajectoryAgent(Agent):

    def __init__(self,
                 network: nn.Module,
                 tau: float,
                 lr: float,
                 grad_clip: float,
                 gamma: float):
        self._network = network
        self._tau = tau
        self._lr = lr
        self._grad_clip = grad_clip
        self._gamma = gamma
        self._nstep = 1

    def build(self, training: bool, device: torch.device = None):
        if device is None:
            device = torch.device('cpu')

        self._q = QFunction(self._network).to(device).train(training)

        if training:
            self._q_target = QFunction(self._network).to(device).train(False)
            utils.soft_updates(self._q, self._q_target, 1.0)

            # Freeze target critic.
            for p in self._q_target.parameters():
                p.requires_grad = False

            self._q_optimizer = torch.optim.Adam(
                list(self._q.parameters()), lr=self._lr, weight_decay=1e-6)

            logging.info('# Trajectory Agent Q Params: %d' % sum(
                p.numel() for p in self._q.parameters() if p.requires_grad))

        self._device = device

    def _grad_step(self, loss, opt, model_params=None, clip=None):
        opt.zero_grad()
        loss.backward()
        if clip is not None and model_params is not None:
            nn.utils.clip_grad_value_(model_params, clip)
        opt.step()

    def _get_tp1_action(self):
        return self._trajectory_tp1

    def update(self, step: int, replay_sample: dict) -> dict:
        reward = replay_sample['reward'] * 0.01
        t = stack_on_channel(replay_sample['low_dim_state'][:, :, -1:])
        tp1 = stack_on_channel(replay_sample['low_dim_state_tp1'][:, :, -1:])
        trajectory_t = stack_on_channel(replay_sample['trajectory'])
        trajectory_tp1 = stack_on_channel(replay_sample['trajectory_tp1'])
        trajectory_t = torch.cat([trajectory_t, t], 1)
        self._trajectory_tp1 = trajectory_tp1 = torch.cat([trajectory_tp1, tp1], 1)
        terminal = replay_sample['terminal'].float()

        q1, q2 = self._q(trajectory_t)
        with torch.no_grad():
            q1_targ, q2_targ = self._q_target(self._get_tp1_action())
            next_value = torch.min(q1_targ, q2_targ)
            q_backup = (reward.unsqueeze(-1) + (
                    self._gamma ** self._nstep) * (1. - terminal.unsqueeze(-1)) * next_value)
            q_backup = torch.clip(q_backup, 0.0, 1.0)

        loss_weights = utils.loss_weights(replay_sample, REPLAY_BETA)
        q1_delta = F.smooth_l1_loss(q1, q_backup, reduction='none').mean(1)
        q2_delta = F.smooth_l1_loss(q2, q_backup, reduction='none').mean(1)

        q1_bellman_loss = (q1_delta * loss_weights).mean()
        q2_bellman_loss = (q2_delta * loss_weights).mean()

        self._critic_summaries = {
            'q1_bellman_loss': q1_bellman_loss,
            'q2_bellman_loss': q2_bellman_loss,
            'q1_mean': q1.mean().item(),
            'q2_mean': q2.mean().item(),
            'traj_t_mean': trajectory_t.mean(),
            'traj_tp1_mean': trajectory_tp1.mean(),
            'traj_t_max': trajectory_t.max(),
            'traj_tp1_max': trajectory_tp1.max(),
        }
        new_pri = torch.sqrt((q1_delta + q2_delta) / 2. + 1e-10)
        new_priority = (new_pri / torch.max(new_pri)).detach()
        self._grad_step(q1_bellman_loss + q2_bellman_loss, self._q_optimizer,
                        list(self._q.parameters()), self._grad_clip)

        utils.soft_updates(self._q, self._q_target, self._tau)
        return {
            'priority': new_priority ** REPLAY_ALPHA
        }

    def act(self, step: int, observation: dict,
            deterministic=False) -> ActResult:
        with torch.no_grad():
            trajectory_t = stack_on_channel(observation['trajectory_t'])
            t = stack_on_channel(
                observation['low_dim_state'][:, :, -1:]).repeat(
                trajectory_t.shape[0], 1)
            trajectory_t = torch.cat([trajectory_t, t], 1)
            q1, q2 = self._q(trajectory_t)
            return ActResult((q1 + q2) / 2)

    def update_summaries(self) -> List[Summary]:
        summaries = []
        for n, v in list(self._critic_summaries.items()):
            summaries.append(ScalarSummary('%s/%s' % (NAME, n), v))
        for tag, param in list(self._q.named_parameters()):
            summaries.append(
                HistogramSummary('%s/gradient/%s' % (NAME, tag), param.grad))
            summaries.append(
                HistogramSummary('%s/weight/%s' % (NAME, tag), param.data))
        return summaries

    def act_summaries(self) -> List[Summary]:
        return []

    def load_weights(self, savedir: str):
        self._q.load_state_dict(
            torch.load(os.path.join(savedir, '%s.pt' % NAME),
                       map_location=self._device))

    def save_weights(self, savedir: str):
        torch.save(
            self._q.state_dict(), os.path.join(savedir, '%s.pt' % NAME))


class TrajectoryWithActorAgent(TrajectoryAgent):

    def __init__(self,
                 network: nn.Module,
                 tau: float,
                 lr: float,
                 grad_clip: float,
                 gamma: float,
                 actor_network: nn.Module,):
        super(TrajectoryWithActorAgent, self).__init__(network, tau, lr, grad_clip, gamma)
        self._actor_network = actor_network

    def build(self, training: bool, device: torch.device = None):
        super(TrajectoryWithActorAgent, self).build(training, device)
        if device is None:
            device = torch.device('cpu')
        self._unkown_col = torch.tensor([UNKNOWN_COLLISION],
                                        device=device, dtype=torch.float32)
        action_min_max = torch.tensor([[[CFG_MIN_ACTIONS]], [[CFG_MAX_ACTIONS]]],
                                      device=device, dtype=torch.float32)
        self._actor = Actor(self._actor_network, action_min_max).to(
            device).train(training)
        if training:
            self._actor_optimizer = torch.optim.Adam(
                list(self._actor.parameters()), lr=self._lr, weight_decay=1e-6)
            logging.info('# Trajectory Agent Pi Params: %d' % sum(
                p.numel() for p in self._actor.parameters() if p.requires_grad))

    def update(self, step: int, replay_sample: dict) -> dict:
        self._replay_samp = replay_sample
        dict = super(TrajectoryWithActorAgent, self).update(step, replay_sample)
        state = stack_on_channel(replay_sample['low_dim_state'])
        traj = stack_on_channel(replay_sample['trajectory'])[:, :-3]
        nbp = stack_on_channel(replay_sample['trajectory'][:, :, -3-7:-3])
        pi = self._actor(torch.cat([state, nbp], -1))

        bc_loss = F.smooth_l1_loss(pi, traj, reduction='none').mean(
            1) * replay_sample['ep_success'].float()

        loss_weights = utils.loss_weights(replay_sample, REPLAY_BETA)
        total_loss = ((bc_loss) * loss_weights).mean()
        self._actor_summaries = {
            'pi/bc_loss': bc_loss.mean(),
            'pi/pi': pi.mean(),
        }
        self._grad_step(total_loss, self._actor_optimizer,
                        self._actor.parameters(), self._grad_clip)
        return dict

    def act_pi(self, step: int, observation: dict,
               deterministic=False) -> ActResult:
        with torch.no_grad():
            state = stack_on_channel(observation['low_dim_state'])
            pi = self._actor(torch.cat([state, observation['nbp']], -1))
            pi = torch.cat([observation['gripper_pose'][:, 0], pi[:, 7:-7],
                            observation['nbp']], -1)
            return ActResult(pi)

    def update_summaries(self) -> List[Summary]:
        summaries = super(TrajectoryWithActorAgent, self).update_summaries()
        for n, v in list(self._actor_summaries.items()):
            summaries.append(ScalarSummary('%s_actor/%s' % (NAME, n), v))
        for tag, param in list(self._actor.named_parameters()):
            summaries.append(
                HistogramSummary('%s_actor/gradient/%s' % (NAME, tag), param.grad))
            summaries.append(
                HistogramSummary('%s_actor/weight/%s' % (NAME, tag), param.data))
        return summaries

    def load_weights(self, savedir: str):
        super(TrajectoryWithActorAgent, self).load_weights(savedir)
        self._actor.load_state_dict(
            torch.load(os.path.join(savedir, '%s_actor.pt' % NAME),
                       map_location=self._device))

    def save_weights(self, savedir: str):
        super(TrajectoryWithActorAgent, self).save_weights(savedir)
        torch.save(
            self._actor.state_dict(),
            os.path.join(savedir, '%s_actor.pt' % NAME))
