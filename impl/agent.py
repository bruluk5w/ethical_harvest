import os
import random
from typing import Union

import numpy as np

from constants import model_name
from impl.config import cfg, get_storage, ensure_folder
from impl.replay_buffer import ReplayBuffer, ReplayFrame


class Agent:
    def __init__(self, initial_state: np.ndarray, actions_ref: np.ndarray, agent_idx: int,  episode_idx=-1):
        self._state_size = initial_state.shape
        self._action_size = actions_ref.shape
        self._frame = 0
        self._replay_buffer = ReplayBuffer(initial_state.shape, cfg().REPLAY_BUFFER_LENGTH,
                                           initial_state.dtype, actions_ref.dtype, np.dtype(np.float32))
        self._last_state = None
        self._last_action = None

        self._agent_idx = agent_idx
        self._episode_idx = episode_idx

        from impl.model import Model

        if self._episode_idx >= 0:
            path_online, path_target = self._make_model_path()
            self._online_network = Model(self._state_size, self._action_size, path_online)
            self._target_network = Model(self._state_size, self._action_size, path_target)
            path_online, path_target = self._make_weights_path()
            self._online_network.load_weights(path_online)
            self._target_network.load_weights(path_target)
        else:
            self._online_network = Model(self._state_size, self._action_size)
            self._target_network = Model(self._state_size, self._action_size)

        # Tracking variables
        self.last_online_q_values = None
        self.last_reward = None
        self.last_explore_probability = None

    @property
    def agent_idx(self):
        return self._agent_idx

    def new_episode(self):
        self._last_state = None
        self._last_action = None
        self._episode_idx += 1

        # Tracking variables
        self.last_online_q_values = None
        self.last_reward = None
        self.last_explore_probability = None

    def save_model(self):
        path_online, path_target = self._make_model_path()
        self._online_network.save(path_online)
        self._target_network.save(path_target)

    def save_weights(self, agent_idx):
        path_online, path_target = self._make_weights_path()
        self._online_network.save_weights(path_online)
        self._target_network.save_weights(path_target)

    def _make_model_path(self):
        filepath = os.path.join(get_storage(), 'models')
        ensure_folder(filepath)
        path_online = os.path.join(filepath, model_name(str(self._agent_idx), "online"))
        path_target = os.path.join(filepath, model_name(str(self._agent_idx), "target"))
        return path_online, path_target

    def _make_weights_path(self):
        filepath = os.path.join(get_storage(), 'weights')
        ensure_folder(filepath)
        path_online = os.path.join(filepath, model_name(str(self._agent_idx), "online", self._episode_idx))
        path_target = os.path.join(filepath, model_name(str(self._agent_idx), "target", self._episode_idx))
        return path_online, path_target

    def step(self, last_reward, new_state, is_terminal_state, training) -> Union[None, int]:
        if new_state is None:
            # we have been shot or are sick
            self._frame += 1
            return None

        action = self._policy(new_state, training)
        if self._last_state is not None:
            self._replay_buffer.remember(ReplayFrame(last_state=self._last_state, last_action=self._last_action,
                                                     reward=last_reward, next_state=new_state,
                                                     is_terminal_state=is_terminal_state))

        if len(self._replay_buffer) >= cfg().REPLAY_BUFFER_MIN_LENGTH_FOR_USE:
            self._train_network()

            update_freq = cfg().TARGET_NETWORK_UPDATE_FREQUENCY
            if update_freq > 0 and not self._frame % update_freq:
                self.update_target_network()

        self._last_state = new_state
        self._last_action = action
        self._frame += 1

        self.last_reward = last_reward

        return action

    def _policy(self, new_state, training=True):
        if training:
            self.last_explore_probability = p_explore = \
                max(cfg().EXPLORE_PROBABILITY_MAX - (self._frame * cfg().EXPLORATION_ANNEALING),
                    cfg().EXPLORE_PROBABILITY_MIN)

        else:
            self.last_explore_probability = p_explore = (
                1.0 if len(self._replay_buffer) <= cfg().REPLAY_BUFFER_MIN_LENGTH_FOR_USE else 0.0)

        if p_explore > random.random():
            self.last_online_q_values = None
            return self._random_action()

        self.last_online_q_values = qvalues = self._online_network(np.expand_dims(new_state, 0))
        return np.squeeze(qvalues).argmax().item()

    def _random_action(self) -> int:
        return int(random.choice(range(self._action_size[0])))

    def _train_network(self):
        last_states, next_states, last_actions, rewards, is_terminal_state = \
            self._replay_buffer.sample(cfg().MINI_BATCH_SIZE)

        next_action = np.argmax(self._online_network(next_states).numpy(), axis=1)
        # the target values of the q values that were chosen in the individual frames
        qs = self._target_network(next_states).numpy()
        target_q_values = (np.squeeze(rewards) + cfg().DISCOUNT_FACTOR *
                           np.squeeze(np.take_along_axis(qs, next_action[None, :].T, 1)))

        # neglect q if next but one state does not exist
        target_q_values[is_terminal_state] = rewards[is_terminal_state]

        num_actions = int(np.prod(self._action_size))
        actions_one_hot = np.eye(num_actions, dtype=np.float32).reshape(num_actions, *self._action_size)[last_actions]

        self._online_network.train_step(last_states, target_q_values, actions_one_hot, is_terminal_state)

    def update_target_network(self):
        self._target_network.copy_variables(self._online_network)
