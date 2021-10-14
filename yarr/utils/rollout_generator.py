from multiprocessing import Value

import numpy as np
from yarr.agents.agent import Agent
from yarr.envs.env import Env
from yarr.utils.transition import ReplayTransition


class RolloutGenerator(object):
    def _get_type(self, x):
        if x.dtype == np.float64:
            return np.float32
        return x.dtype

    def generator(self, step_signal: Value, env: Env, agent: Agent,
                  episode_length: int, timesteps: int, eval: bool):
        obs = env.reset()
        agent.reset()
        obs_history = {
            k: [np.array(v, dtype=self._get_type(v))] * timesteps
            for k, v in obs.items()
        }
        for step in range(episode_length):

            prepped_data = {k: np.array([v]) for k, v in obs_history.items()}

            act_result = agent.act(step_signal.value,
                                   prepped_data,
                                   deterministic=eval)

            # Convert to np if not already
            agent_obs_elems = {
                k: np.array(v)
                for k, v in act_result.observation_elements.items()
            }
            extra_replay_elements = {
                k: np.array(v)
                for k, v in act_result.replay_elements.items()
            }

            transition = env.step(act_result)
            obs_tp1 = dict(transition.observation)
            timeout = False
            if step == episode_length - 1:
                # If last transition, and not terminal, then we timed out
                timeout = not transition.terminal
                if timeout:
                    transition.terminal = True
                    if "needs_reset" in transition.info:
                        transition.info["needs_reset"] = True

            obs_and_replay_elems = {}
            obs_and_replay_elems.update(obs)
            obs_and_replay_elems.update(agent_obs_elems)
            obs_and_replay_elems.update(extra_replay_elements)

            for k in obs_history.keys():
                obs_history[k].append(transition.observation[k])
                obs_history[k].pop(0)

            replay_transition = ReplayTransition(
                obs_and_replay_elems,
                act_result.action,
                transition.reward,
                transition.terminal,
                timeout,
                summaries=transition.summaries)

            if transition.terminal or timeout:
                # If the agent gives us observations then we need to call act
                # one last time (i.e. acting in the terminal state).
                if len(act_result.observation_elements) > 0:
                    prepped_data = {
                        k: np.array([v])
                        for k, v in obs_history.items()
                    }
                    act_result = agent.act(step_signal.value,
                                           prepped_data,
                                           deterministic=eval)
                    agent_obs_elems_tp1 = {
                        k: np.array(v)
                        for k, v in act_result.observation_elements.items()
                    }
                    obs_tp1.update(agent_obs_elems_tp1)
                replay_transition.final_observation = obs_tp1

            obs = dict(transition.observation)
            yield replay_transition

            if transition.info.get("needs_reset", transition.terminal):
                return
