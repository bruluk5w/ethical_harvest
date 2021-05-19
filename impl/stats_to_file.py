import time
from threading import Thread
from multiprocessing import Condition
from collections import defaultdict, deque
from typing import Dict, List

import numpy as np

from impl.config import get_storage
import os.path
from impl.stats import Stats, AgentSeries, Summary
from envs.env import PlayerSprite, AppleDrape

_file_conditions = defaultdict((lambda: Condition()))  # type: Dict[str, Condition]


def get_trace_file_path(base_path=None):
    if base_path is None:
        base_path = get_storage()

    return os.path.join(base_path, 'trace.txt')


class StatsWriter:

    def __init__(self, file_path=None):
        self._file_path = file_path
        self._file = None if self._file_path is None else open(self._file_path, 'w', encoding='utf-8')
        self.frames = 0

    def __del__(self):
        if self._file is not None and not self._file.closed:
            self._file.close()

    def on_episode_frame(self, agents, avatars: List[PlayerSprite], apple_drape: AppleDrape, episode_idx):
        for agent, avatar in zip(sorted(agents, key=lambda a: a.agent_idx), avatars):
            used_network = agent.last_online_q_values is not None
            self._handle_agent_step_data(
                agent.agent_idx,
                agent.last_online_q_values.numpy().max() if used_network else None,
                agent.last_reward,
                agent.last_explore_probability,
                avatar.has_apples,
                avatar.donated_apples,
                avatar.taken_apples,
            )

        self._handle_frame_data(
            apple_drape.common_pool,
            apple_drape.curtain[5:, :].sum(axis=None).item(),
        )

        self.frames += 1

    def on_episode_end(self, agents, episode_idx):
        if self._file:
            condition = _file_conditions[self._file_path]
            with condition:
                self._file.write("Episode End: {},{}\n".format(episode_idx, self.frames))
                self._file.flush()
                condition.notify_all()

    def _handle_agent_step_data(self, agent_idx, last_q, last_reward, last_explore_probability,
                                num_owned_apples, num_donated_apples, num_taken_donations):
        if self._file:
            condition = _file_conditions[self._file_path]
            with condition:
                self._file.write("{},{},{},{},{},{},{}\n".format(
                    agent_idx, last_q, last_reward, last_explore_probability,
                    num_owned_apples, num_donated_apples, num_taken_donations))
                # self._file.flush()
                # condition.notify_all()

    def _handle_frame_data(self, num_common_pool, num_free_apples):
        if self._file:
            condition = _file_conditions[self._file_path]
            with condition:
                self._file.write("Frame: {},{}\n".format(num_common_pool, num_free_apples))
                # self._file.flush()
                # condition.notify_all()


class QStatsReader(Thread):

    def __init__(self, file_path=None, callback=None):
        super().__init__()
        self._file_path = file_path
        self._callback = callback
        self._file = None if self._file_path is None else open(self._file_path, 'r', encoding='utf-8')
        self._stopping = False
        self._reached_end = False
        self._stats = Stats([], deque(), deque(), Summary([], [], [], [], [], [], [], [], []), deque())

        self._num_frames = 0

        if self._file is not None:
            self.start()

    def __del__(self):
        if self._initialized:
            self.stop()

    def stop(self):
        if self.is_alive():
            self._stopping = True
            condition = _file_conditions[self._file_path]
            with condition:
                condition.notify_all()

            self.join()

        if self._file and not self._file.closed:
            self._file.close()

    def run(self):
        condition = _file_conditions[self._file_path]
        with condition:
            while not self._stopping:
                self._ingest_data()
                if self._reached_end:
                    if self._callback is not None and self.has_data:
                        self._callback(self._stats)
                    condition.wait()
                else:
                    if self._callback is not None:
                        self._callback(self._stats)
                    # give other thread a chance
                    time.sleep(0.001)

    @property
    def has_data(self) -> bool:
        return len(self._stats.episode_ends) > 0

    def _ingest_data(self):
        if self._file:
            if self._reached_end:
                for line in self._file.readlines():
                    self._handle_line(line)
            else:
                for line in self._file.readlines(1024*1024):
                    self._handle_line(line)

                line = self._file.readline()
                if line == '':
                    self._reached_end = True
                else:
                    self._handle_line(line)

    def _handle_line(self, line):
        if line.startswith('Episode End: '):
            self._read_episode_end(line[13:])
        elif line.startswith('Frame: '):
            self._read_episode_frame(line[7:])
        else:
            self._read_agent_frame(line)

    def _read_episode_frame(self, line):
        num_common_pool, num_free_apples = tuple(line.rstrip('\n').split(','))
        self._stats.num_common_pool.append(int(num_common_pool))
        self._stats.num_free_apples.append(int(num_free_apples))
        self._num_frames += 1

    def _read_agent_frame(self, line):
        agent_idx, last_q, last_reward, last_explore_probability, num_owned_apples, num_donated_apples, \
        num_taken_donations = tuple(line.rstrip('\n').split(','))

        agent_idx = int(agent_idx)
        last_q = float('nan') if last_q == 'None' else float(last_q)
        last_reward = float('nan') if last_reward == 'None' else float(last_reward)
        last_explore_probability = float('nan') if last_explore_probability == 'None' else float(last_explore_probability)
        num_owned_apples = int(num_owned_apples)
        num_donated_apples = int(num_donated_apples)
        num_taken_donations = int(num_taken_donations)

        if agent_idx >= len(self._stats.agent_series):
            self._stats.agent_series.extend(AgentSeries(last_q=deque([float('nan')] * self._num_frames),
                                                        last_reward=deque([float('nan')] * self._num_frames),
                                                        last_explore_probability=deque([float('nan')] * self._num_frames),
                                                        num_owned_apples=deque([0] * self._num_frames),
                                                        num_donated_apples=deque([0] * self._num_frames),
                                                        num_taken_donations=deque([0] * self._num_frames),
                                                        summary=Summary([], [], [], [], [], [], [], [], []))
                                            for _ in range(agent_idx - len(self._stats.agent_series) + 1))

        series = self._stats.agent_series[agent_idx]
        series.last_q.append(last_q)
        series.last_reward.append(last_reward)
        series.last_explore_probability.append(last_explore_probability)
        series.num_owned_apples.append(num_owned_apples)
        series.num_donated_apples.append(num_donated_apples)
        series.num_taken_donations.append(num_taken_donations)

    def _read_episode_end(self, line):
        episode_idx, episode_num_frames = tuple(line.rstrip('\n').split(','))
        episode_idx = int(episode_idx)
        episode_num_frames = int(episode_num_frames)
        self._stats.episode_ends.append(episode_num_frames)
        if self._callback is not None and self._reached_end:
            self._callback(self._stats)
        # self.max_q = float('-inf')
        # self.min_q = float('+inf')
        # self.sum_q = 0
        #
        # self.max_q, self.min_q = max(self.max_q, last_q), min(self.min_q, last_q)
        # self.sum_q += last_q
