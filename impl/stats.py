from enum import Enum
from typing import NamedTuple, List, Union

import numpy as np


class Properties(Enum):
    LAST_Q = 'last_q'
    LAST_REWARD = 'last_reward'
    LAST_EXPLORE_PROBABILITY = 'last_explore_probability'


class SummaryProperties(Enum):
    # per episode aggregates
    MIN_Q = 'min_q'
    MAX_Q = 'max_q'
    AVG_Q = 'avg_q'
    SUM_REWARD = 'sum_reward'
    EPISODE_START = 'episode_start'


Summary = NamedTuple('Summary', [
    ('max_q', List[float]),
    ('min_q', List[float]),
    ('avg_q', List[float]),
    ('episode_start', List[int]),
])

AgentSeries = NamedTuple('AgentSeries', [
    ('last_q', List[Union[float, None]]),
    ('last_reward', List[Union[float, None]]),
    ('last_explore_probability', List[Union[float, None]]),
    ('summary', Union[Summary, None]),
])


Stats = NamedTuple('Stats', [
    ('agent_series', List[AgentSeries]),
    ('summary', Union[Summary, None]),
    ('episode_ends', List[int]),
])


_default_values = {
    float: float('nan')
}


#df[''] = df[''].rolling(2, win_type=None).mean()


def build_summary(stats: Stats):
    max_frames = stats.episode_ends[-1] if len(stats.episode_ends) > 0 else 0
    # make all series the same length
    for agent_series in stats.agent_series:
        for series, t in ((getattr(agent_series, attr), t) for attr, t in agent_series._field_types.items()):
            if t.__origin__ is List:
                default_t = t.__args__[0]
                default_value = _default_values.get(default_t, None)
                if default_value is None:
                    default_value = default_t()

                series.extend(default_value for _ in range(max_frames - len(series) + 1))

    for idx, agent_series in enumerate(stats.agent_series):
        summary = agent_series.summary
        episode_start = 0
        for episode_end in stats.episode_ends:
            summary.max_q.append(max(agent_series.last_q[episode_start:episode_end]))
            summary.min_q.append(min(agent_series.last_q[episode_start:episode_end]))
            summary.avg_q.append(sum(agent_series.last_q[episode_start:episode_end]) / (episode_end - episode_start))
            summary.episode_start.append(episode_start)
            episode_start = episode_end

        summary.episode_start.append(episode_start)

        stats.agent_series[idx] = summary._replace(
            max_q=np.array(summary.max_q),
            min_q=np.array(summary.max_q),
            avg_q=np.array(summary.max_q),
            episode_start=np.array(summary.max_q),
        )

    stats = stats._replace(
        summary=Summary(
            max_q=stats.agent_series[0].summary.max_q.copy(),
            min_q=stats.agent_series[0].summary.min_q.copy(),
            avg_q=stats.agent_series[0].summary.avg_q.copy(),
            episode_start=stats.agent_series[0].summary.episode_start.copy(),
        )
    )
    summary = stats.summary
    for agent_series in stats.agent_series[1:]:
        summary.max_q += agent_series.summary.max_q
        summary.min_q += agent_series.summary.min_q
        summary.avg_q += agent_series.summary.avg_q

    num_agents = len(stats.agent_series)
    summary.max_q /= num_agents
    summary.min_q /= num_agents
    summary.avg_q /= num_agents
