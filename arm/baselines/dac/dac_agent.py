import copy
import logging
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from yarr.agents.agent import Summary, ScalarSummary, HistogramSummary

from arm import utils
from arm.baselines.sac import sac_agent
from arm.baselines.sac.sac_agent import SACAgent
from arm.utils import stack_on_channel

NAME = 'DSACAgent'

REPLAY_ALPHA = 0.7
REPLAY_BETA = 1.0


class Discriminator(nn.Module):

    def __init__(self, discrim_net: nn.Module):
        super(Discriminator, self).__init__()
        self._discrim_net = copy.deepcopy(discrim_net)
        self._discrim_net.build()

    def forward(self, observations, robot_state, action):
        # Assume first 50% are demos, rest are explore data
        combined = torch.cat([robot_state, action.float()], dim=1)
        return self._discrim_net(observations, combined)


class DACAgent(SACAgent):

    def __init__(self,
                 discriminator_network: nn.Module,
                 lambda_gp: float,
                 discriminator_lr: float,
                 discriminator_grad_clip: float,
                 discriminator_weight_decay: float,
                 **kwargs):
        super(DACAgent, self).__init__(**kwargs)
        self._discriminator_network = discriminator_network
        self._lambda_gp = lambda_gp
        self._discriminator_lr = discriminator_lr
        self._discriminator_grad_clip = discriminator_grad_clip
        self._discriminator_weight_decay = discriminator_weight_decay
        sac_agent.NAME = NAME

    def build(self, training: bool, device: torch.device = None):
        super(DACAgent, self).build(training, device)
        if device is None:
            device = torch.device('cpu')
        if training:
            self._discrim = Discriminator(self._discriminator_network).to(
                device).train(training)
            self._discrim_optimizer = torch.optim.Adam(
                self._discrim.parameters(), lr=self._discriminator_lr,
                weight_decay=self._discriminator_weight_decay)
            logging.info('# Discriminator Params: %d' % sum(
                p.numel() for p in self._discrim.parameters() if p.requires_grad))
        self._device = device

    def _interp(self, expert, fake, alpha):
        return torch.autograd.Variable(
            alpha * expert + (1 - alpha) * fake, requires_grad=True)

    def _gan_loss(self, observations, robot_state, action):
        # Assume first 50% are demos, rest are explore data
        b = action.shape[0]
        alpha = torch.rand((b//2, 1), device=action.device)
        inter_robot_state = self._interp(
            robot_state[:b // 2], robot_state[b // 2:], alpha)
        inter_action = self._interp(action[:b // 2], action[b // 2:], alpha)
        alpha_img = alpha.unsqueeze(-1).unsqueeze(-1)
        inter_observations = [self._interp(
            o[:b // 2], o[b // 2:], alpha_img) for o in observations]

        experts, fakes = torch.split(self._discrim(observations, robot_state, action), b//2, dim=0)
        inter_out = self._discrim(inter_observations, inter_robot_state, inter_action)

        inter_observations.append(inter_action)
        inter_observations.append(inter_robot_state)
        gradient = torch.autograd.grad(
            inputs=inter_observations,
            outputs=inter_out,
            grad_outputs=torch.ones_like(inter_out),
            create_graph=True,
            retain_graph=True
        )[0]
        gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean() * self._lambda_gp

        expert_loss = F.binary_cross_entropy_with_logits(
            experts, torch.ones_like(experts), reduction='none')
        fakes_loss = F.binary_cross_entropy_with_logits(
            fakes, torch.zeros_like(fakes), reduction='none')

        pri = torch.sqrt(torch.cat([expert_loss, fakes_loss], 0).mean(1) / 2. + 1e-10)
        pri = (pri / torch.max(pri)).detach()

        return (expert_loss + fakes_loss).mean() + gradient_penalty, gradient_penalty, experts, fakes, pri


    def update(self, step: int, replay_sample: dict) -> dict:

        robot_state = stack_on_channel(replay_sample['low_dim_state'][:, -1:])
        observations = [
            stack_on_channel(replay_sample['%s_rgb' % self._camera_name]),
            stack_on_channel(
                replay_sample['%s_point_cloud' % self._camera_name])
        ]

        # Update GAN
        gan_loss, grad_pen, experts, fakes, pri = self._gan_loss(
            observations, robot_state, replay_sample['action'])
        self._grad_step(
            gan_loss, self._discrim_optimizer, self._discrim.parameters(),
            self._discriminator_grad_clip)

        # Only use explore data for RL update.
        b = robot_state.shape[0]
        rl_replay_sample = {k: v[b//2:] for k, v in replay_sample.items()}
        with torch.no_grad():
            reward = self._discrim([o[b//2:] for o in observations], robot_state[b//2:], rl_replay_sample['action'])
            reward = -torch.log(1 - torch.sigmoid(reward) + 1e-8)
        self._update_critic(rl_replay_sample, reward[:, 0])

        # Freeze critic so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self._q.parameters():
            p.requires_grad = False

        if step % 2 == 0:
            self._update_actor(rl_replay_sample)
            if self._alpha_auto_tune:
                self._update_alpha()

        # UnFreeze critic.
        for p in self._q.parameters():
            p.requires_grad = True

        decoder_pri = self._update_decoder(replay_sample)
        pri += decoder_pri
        pri = torch.ones_like(pri)
        self._new_priority = torch.cat([torch.zeros_like(self._new_priority), self._new_priority])
        self._new_priority += pri

        if step % 2 == 0:
            utils.soft_updates(self._q._q1, self._q_target._q1, self._critic_tau)
            utils.soft_updates(self._q._q2, self._q_target._q2, self._critic_tau)
            utils.soft_updates(self._q.encoder, self._q_target.encoder, self._encoder_tau)

        self._gan_summaries = {
            'gan/loss': gan_loss,
            'gan/gradient_penalty': grad_pen,
            'gan/expert_accuracy': torch.sigmoid(experts).mean(),
            'gan/fake_accuracy': 1. - torch.sigmoid(fakes).mean(),
        }
        return {
            # 'priority': self._new_priority ** REPLAY_ALPHA
            'priority': pri ** REPLAY_ALPHA
        }

    def update_summaries(self) -> List[Summary]:
        summaries = super(DACAgent, self).update_summaries()
        for n, v in self._gan_summaries.items():
            summaries.append(ScalarSummary('%s/%s' % (NAME, n), v))
        for tag, param in self._discrim.named_parameters():
            summaries.append(
                HistogramSummary('%s/gradient/%s' % (NAME, tag),
                                 param.grad))
            summaries.append(
                HistogramSummary('%s/weight/%s' % (NAME, tag), param.data))
        return summaries
