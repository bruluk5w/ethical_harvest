import os
import random
from typing import Tuple, Union

import numpy as np

from impl.model import Model
from impl.replay_buffer import ReplayBuffer, ReplayFrame
from impl.config import cfg, get_storage, ensure_folder


class Agent:
    def __init__(self, initial_state: np.ndarray, actions_ref: np.ndarray):
        self._state_size = initial_state.shape
        self._action_size = actions_ref.shape
        self._online_network = Model(self._state_size, self._action_size)
        self._target_network = Model(self._state_size, self._action_size)
        self._frame = 0
        self._replay_buffer = ReplayBuffer(initial_state.shape, cfg().REPLAY_BUFFER_LENGTH,
                                           initial_state.dtype, actions_ref.dtype, np.dtype(np.float32))
        self._last_state = None
        self._last_action = None
        self._episode_idx = -1

    def new_episode(self):
        self._last_state = None
        self._last_action = None
        self._episode_idx += 1

    def save(self, agent_idx):
        filepath = os.path.join(get_storage(), 'models')
        ensure_folder(filepath)
        path_online = os.path.join(filepath, 'episode_{}_agent_{}_online'.format(self._episode_idx, str(agent_idx)))
        path_target = os.path.join(filepath, 'episode_{}_agent_{}_target'.format(self._episode_idx, str(agent_idx)))
        self._online_network.save(path_online)
        self._target_network.save(path_target)

    def step(self, last_reward, new_state, training) -> Union[None, Tuple[int]]:
        if new_state is None:
            # we have been shot or are sick
            self._frame += 1
            return None

        if not training:
            action = self._random_action()
        else:
            action = self._epsilon_greedy_action(new_state)
            if self._last_state is not None:
                self._replay_buffer.remember(ReplayFrame(last_state=self._last_state, last_action=self._last_action,
                                                         reward=last_reward, next_state=new_state))

            if len(self._replay_buffer) >= cfg().REPLAY_BUFFER_MIN_LENGTH_FOR_USE:
                self._train_network()

                if not self._frame % cfg().TARGET_NETWORK_UPDATE_FREQUENCY:
                    self._update_target_network()

        self._last_state = new_state
        self._last_action = action
        self._frame += 1

        return action

    def _epsilon_greedy_action(self, new_state):
        p_explore = max(cfg().EXPLORE_PROBABILITY_MAX - (self._frame * cfg().EXPLORATION_ANNEALING),
                        cfg().EXPLORE_PROBABILITY_MIN)
        if p_explore > random.random():
            return self._random_action()
        else:
            qvalues = self._online_network(np.expand_dims(new_state, 0))
            #  todo: fix: 2d action index
            return np.squeeze(qvalues).argmax(),

    def _random_action(self) -> Tuple[int]:
        return tuple(int(random.choice(range(size))) for size in self._action_size)

    def _train_network(self):
        states, next_states, actions, rewards = self._replay_buffer.sample(cfg().MINI_BATCH_SIZE)

        next_qvalues = self._target_network(next_states)
        # the target values of the q values that were chosen in the individual frames
        target_q_values = np.squeeze(rewards) + cfg().DISCOUNT_FACTOR * np.amax(next_qvalues, axis=-1)

        num_actions = int(np.prod(self._action_size))
        actions_one_hot = np.eye(num_actions, dtype=np.float32).reshape(num_actions, *self._action_size)[actions]

        self._online_network.train_step((states, target_q_values, actions_one_hot))

    def _update_target_network(self):
        self._target_network.copy_variables(self._online_network)
