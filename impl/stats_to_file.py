from threading import Thread
from multiprocessing import Condition
from collections import defaultdict
from typing import Dict

from impl.config import get_storage
import os.path
from impl.stats import Stats, AgentSeries, Summary

_file_conditions = defaultdict((lambda: Condition()))  # type: Dict[str, Condition]


def get_trace_file_path(base_path=None):
    if base_path is None:
        base_path = get_storage()

    return os.path.join(base_path, 'trace.txt')


class QStatsDumper:

    def __init__(self, file_path=None):
        self._file_path = file_path
        self._file = None if self._file_path is None else open(self._file_path, 'w', encoding='utf-8')
        self.frames = 0

    def __del__(self):
        if self._file is not None and not self._file.closed:
            self._file.close()

    def on_episode_frame(self, agents, episode_idx):
        for agent in sorted(agents, key=lambda a: a.agent_idx):
            used_network = agent.last_online_q_values is not None
            self._handle_step_agent_data(
                agent.agent_idx,
                agent.last_online_q_values.numpy().max() if used_network else None,
                agent.last_reward,
                agent.last_explore_probability
            )

        self.frames += 1

    def on_episode_end(self, agents, episode_idx):
        self._handle_episode_end(episode_idx)

    def _handle_step_agent_data(self, agent_idx, last_q, last_reward, last_explore_probability):
        if self._file:
            condition = _file_conditions[self._file_path]
            with condition:
                self._file.write("{},{},{},{}\n".format(agent_idx, last_q, last_reward, last_explore_probability))
                self._file.flush()
                condition.notify_all()

    def _handle_episode_end(self, episode_idx):
        if self._file:
            condition = _file_conditions[self._file_path]
            with condition:
                self._file.write("Episode End: {},{}\n".format(episode_idx, self.frames))
                self._file.flush()
                condition.notify_all()


class QStatsReader(Thread):

    def __init__(self, file_path=None, callback=None, ):
        super().__init__()
        self._file_path = file_path
        self._callback = callback
        self._file = None if self._file_path is None else open(self._file_path, 'r', encoding='utf-8')
        self._stopping = False
        self._reached_end = False
        self._stats = Stats([], Summary([], [], [], [], []), [])

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
                    condition.wait()

    def _ingest_data(self):
        if self._file:
            if self._reached_end:
                for line in self._file.readlines():
                    self._handle_line(line)
            else:
                for line in self._file.readlines(1024):
                    self._handle_line(line)

                line = self._file.readline()
                if line == '':
                    self._reached_end = True
                else:
                    self._handle_line(line)

    def _handle_line(self, line):
        if line.startswith('Episode End: '):
            self._read_episode_end(line[13:])
        else:
            self._read_episode_frame(line)

    def _read_episode_frame(self, line):
        agent_idx, last_q, last_reward, last_explore_probability = tuple(line.rstrip('\n').split(','))
        agent_idx = int(agent_idx)
        last_q = float('nan') if last_q == 'None' else float(last_q)
        last_reward = float('nan') if last_reward == 'None' else float(last_reward)
        last_explore_probability = float('nan') if last_explore_probability == 'None' else float(last_explore_probability)

        if agent_idx >= len(self._stats.agent_series):
            self._stats.agent_series.extend(AgentSeries(last_q=[float('nan')] * self._num_frames,
                                                        last_reward=[float('nan')] * self._num_frames,
                                                        last_explore_probability=[float('nan')] * self._num_frames,
                                                        summary=Summary([], [], [], [], []))
                                            for _ in range(agent_idx - len(self._stats.agent_series) + 1))

        series = self._stats.agent_series[agent_idx]
        series.last_q.append(last_q)
        series.last_reward.append(last_reward)
        series.last_explore_probability.append(last_explore_probability)

        self._num_frames += 1

    def _read_episode_end(self, line):
        episode_idx, episode_num_frames = tuple(line.rstrip('\n').split(','))
        episode_idx = int(episode_idx)
        episode_num_frames = int(episode_num_frames)
        self._stats.episode_ends.append(episode_num_frames)
        if self._callback is not None:
            self._callback(self._stats)
        # self.max_q = float('-inf')
        # self.min_q = float('+inf')
        # self.sum_q = 0
        #
        # self.max_q, self.min_q = max(self.max_q, last_q), min(self.min_q, last_q)
        # self.sum_q += last_q
