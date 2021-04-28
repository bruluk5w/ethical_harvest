from collections import namedtuple
from random import sample
from typing import Tuple

import numpy as np


class ReplayFrame(namedtuple('ReplayFrame', ['last_state', 'last_action', 'reward', 'next_state'])):
    __slots__ = ()

    def __str__(self):
        return super().__str__()


class ReplayBuffer:
    def __init__(self, state_size: Tuple, length: int, state_dtype: np.dtype, action_dtype: np.dtype, reward_dtype: np.dtype):
        self._length = 0
        self._cursor = 0
        self._state_buffer = np.ndarray((length,) + state_size, dtype=state_dtype)
        self._next_state_buffer = np.empty_like(self._state_buffer)
        self._action_buffer = np.ndarray((length, 1), dtype=action_dtype)
        self._reward_buffer = np.ndarray((length, 1), dtype=reward_dtype)

    def __len__(self):
        return self._length

    def sample(self, batch_length):
        batch_length = min(self._length, batch_length)
        assert self._length >= min(self._length, batch_length)
        samples = sample(range(self._length), batch_length)
        return (self._state_buffer[samples], self._next_state_buffer[samples],
                self._action_buffer[samples], self._reward_buffer[samples])

    def remember(self, memory_frame: ReplayFrame):
        self._state_buffer[self._cursor] = memory_frame.last_state
        self._action_buffer[self._cursor] = memory_frame.last_action
        self._reward_buffer[self._cursor] = memory_frame.reward
        self._next_state_buffer[self._cursor] = memory_frame.next_state

        if self._length < self._state_buffer.shape[0]:
            self._length += 1

        self._cursor = self._cursor + 1
        if self._length == self._state_buffer.shape[0]:
            self._cursor = self._cursor % self._length
