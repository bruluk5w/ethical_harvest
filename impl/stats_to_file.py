from threading import Thread
from multiprocessing import Condition
from collections import defaultdict
from typing import Dict, NamedTuple, List, Union
from config import cfg, get_storage
import os.path

_file_conditions = defaultdict((lambda x: Condition()))  # type: Dict[str, Condition]


def get_trace_file_path():
    return os.path.join(get_storage(), 'trace.txt')


class QStatsDumper:

    def __init__(self, file_path=None):
        self._file_path = file_path
        self._file = None if self._file_path is None else open(self._file_path, 'w')
        self.frames = 0

    def __del__(self):
        if self._file is not None and not self._file.closed:
            self._file.close()

    def on_episode_frame(self, agents, episode_idx):
        for agent in sorted(agents, key=lambda a: a.agent_idx):
            was_active = agent.last_online_q_values is not None
            self._handle_step_agent_data(
                agent.agent_idx,
                agent.last_online_q_values.numpy().max() if was_active else None,
                agent.last_reward
            )

        self.frames += 1

    def on_episode_end(self, agents, episode_idx):
        self._handle_episode_end(episode_idx)

    def _handle_step_agent_data(self, agent_idx, last_q, last_reward):
        if self._file:
            with _file_conditions[self._file_path] as c:
                self._file.write("{},{},{}".format(agent_idx, last_q, last_reward))
                self._file.flush()
                c.notify_all()

    def _handle_episode_end(self, episode_idx):
        if self._file:
            with _file_conditions[self._file_path] as c:
                self._file.write("Episode End: {},{}".format(episode_idx, self.frames))
                self._file.flush()
                c.notify_all()


AgentSeries = NamedTuple('AgentSeries', [
    ('last_q', List[Union[None, float]]),
    ('last_reward', List[Union[None, float]]),

])

Stats = NamedTuple('Stats', [
    ("agent_series", List[AgentSeries]),
    ("episode_ends", List[int])
])


class InvalidTraceException(Exception):
    pass


class QStatsReader(Thread):

    def __init__(self, file_path=None, callback=None, update_per_episode=True):
        super().__init__()
        self._file_path = file_path
        self._callback = callback
        self._update_per_episode = update_per_episode
        self._file = None if self._file_path is None else open(self._file_path, 'r')
        self._stopping = False
        self._stats = Stats(agent_series=[], episode_ends=[])

        self._num_frames = 0

        if self._file is not None:
            self.start()

    def __del__(self):
        self.stop()

    def stop(self):
        if self.is_alive():
            self._stopping = True
            with _file_conditions[self._file_path] as c:
                c.notify_all()

            self.join()

        if self._file and not self._file.closed:
            self._file.close()

    def run(self):
        with _file_conditions[self._file_path] as c:
            while not self._stopping:
                self._ingest_data()
                c.wait()

    def _ingest_data(self):
        if self._file:
            line = self._file.readline()
            if line.startswith('Episode End: '):
                self._read_episode_end(line[13:])
            else:
                self._read_episode_frame(line)

    def _read_episode_frame(self, line):
        agent_idx, last_q, last_reward = line.split(',')
        agent_idx = int(agent_idx)
        last_q = float('nan') if last_q == 'None' else float(last_q)
        last_reward = float('nan') if last_q == 'None' else float(last_reward)

        if agent_idx >= len(self._stats.agent_series):
            self._agents.extend(AgentSeries(last_q=[float('nan')] * self._num_frames,
                                            last_reward=[float('nan')] * self._num_frames)
                                for _ in range(agent_idx - len(self._agents)))

        series = self._stats.agent_series[agent_idx]
        series.last_q.append(last_q)
        series.last_reward.append(last_reward)

        self._num_frames += 1

        if self._callback is not None and not self._update_per_episode:
            self._callback(self._stats)

    def _read_episode_end(self, line):
        episode_idx, episode_num_frames = line.split(',')
        if len(self._stats.episode_ends) > 0 and self._stats.episode_ends[-1] - self._num_frames != episode_num_frames:
            raise InvalidTraceException('Inconsistent number of frames for this episode')

        self._stats.episode_ends.append(self._num_frames)
        if self._callback is not None:
            self._callback(self._stats)
        # self.max_q = float('-inf')
        # self.min_q = float('+inf')
        # self.sum_q = 0
        #
        # self.max_q, self.min_q = max(self.max_q, last_q), min(self.min_q, last_q)
        # self.sum_q += last_q
