import typing
from enum import Enum
from typing import NamedTuple, List, Union
import numpy as np


class Properties(Enum):
    LAST_Q = 'last_q'
    LAST_REWARD = 'last_reward'
    LAST_EXPLORE_PROBABILITY = 'last_explore_probability'
    NUM_OWNED_APPLES = 'num_owned_apples'
    NUM_DONATED_APPLES = 'num_donated_apples'
    NUM_TAKEN_DONATIONS = 'num_taken_donations'


class SummaryProperties(Enum):
    # per episode aggregates
    MIN_Q = 'min_q'
    MAX_Q = 'max_q'
    AVG_Q = 'avg_q'
    SUM_REWARD = 'sum_reward'
    MIN_OWNED_APPLES = 'min_owned_apples'
    MAX_OWNED_APPLES = 'max_owned_apples'
    AVG_OWNED_APPLES = 'avg_owned_apples'
    TOTAL_DONATED_APPLES = 'total_donated_apples'
    EPISODE_START = 'episode_start'


Summary = NamedTuple('Summary', [
    ('max_q', Union[List[float], np.ndarray]),
    ('min_q', Union[List[float], np.ndarray]),
    ('avg_q', Union[List[float], np.ndarray]),
    ('sum_reward', Union[List[float], np.ndarray]),
    ('min_owned_apples',  Union[List[int], np.ndarray]),
    ('max_owned_apples',  Union[List[int], np.ndarray]),
    ('avg_owned_apples',  Union[List[int], np.ndarray]),
    ('total_donated_apples', Union[List[int], np.ndarray]),
    ('episode_start', Union[List[int], np.ndarray]),
])

AgentSeries = NamedTuple('AgentSeries', [
    ('last_q', List[Union[float, None]]),
    ('last_reward', List[Union[float, None]]),
    ('last_explore_probability', List[Union[float, None]]),
    ('num_owned_apples', List[int]),
    ('num_donated_apples', List[int]),
    ('num_taken_donations', List[int]),
    ('summary', Union[Summary, None]),
])


Stats = NamedTuple('Stats', [
    ('agent_series', List[AgentSeries]),
    ('num_common_pool', List[int]),
    ('num_free_apples', List[int]),
    ('summary', Union[Summary, None]),
    ('episode_ends', List[int]),
])


_default_values = {
    float: float('nan')
}


def init_np_copy(stats: Stats):
    np_stats = Stats(
        agent_series=[None] * len(stats.agent_series),
        num_common_pool=np.array(stats.num_common_pool),
        num_free_apples=np.array(stats.num_free_apples),
        summary=None,
        episode_ends=np.array(stats.episode_ends)
    )

    for idx, agent_series in enumerate(stats.agent_series):
        # convert to numpy and make arrays the same length
        max_frames = stats.episode_ends[-1] if len(np_stats.episode_ends) > 0 else 0

        last_q = np.ma.masked_invalid(agent_series.last_q)
        if last_q.shape[0] < max_frames:
            last_q = np.concatenate([last_q, [float('nan') for _ in range(max_frames - len(last_q))]])
        last_reward = np.ma.masked_invalid(agent_series.last_reward)
        if last_reward.shape[0] < max_frames:
            last_reward = np.concatenate([last_reward, [float('nan') for _ in range(max_frames - len(last_reward))]])
        last_explore = np.ma.masked_invalid(agent_series.last_explore_probability)
        if last_explore.shape[0] < max_frames:
            last_explore = np.concatenate([last_explore, [float('nan') for _ in range(max_frames - len(last_explore))]])
        num_owned_apples = np.array(agent_series.num_owned_apples)
        if num_owned_apples.shape[0] < max_frames:
            num_owned_apples = np.concatenate([num_owned_apples,
                                               [(num_owned_apples[-1] if num_owned_apples.shape[0] > 0 else 0) for _ in
                                                range(max_frames - len(num_owned_apples))]])
        num_donated_apples = np.array(agent_series.num_donated_apples)
        if num_donated_apples.shape[0] < max_frames:
            num_donated_apples = np.concatenate([num_donated_apples,
                                                 [(num_donated_apples[-1] if num_donated_apples.shape[0] > 0 else 0) for _ in
                                                  range(max_frames - len(num_donated_apples))]])
        num_taken_donations = np.array(agent_series.num_taken_donations)
        if num_taken_donations.shape[0] < max_frames:
            num_taken_donations = np.concatenate([num_taken_donations,
                                                  [(num_taken_donations[-1] if num_taken_donations.shape[0] > 0 else 0) for _ in
                                                   range(max_frames - len(num_taken_donations))]])

        num_episodes = np_stats.episode_ends.shape[0]
        np_stats.agent_series[idx] = AgentSeries(
            last_q=last_q,
            last_reward=last_reward,
            last_explore_probability=last_explore,
            num_owned_apples=num_owned_apples,
            num_donated_apples=num_donated_apples,
            num_taken_donations=num_taken_donations,
            summary=Summary(
                max_q=np.empty((num_episodes,), dtype=last_q.dtype),
                min_q=np.empty((num_episodes,), dtype=last_q.dtype),
                avg_q=np.empty((num_episodes,), dtype=np.float32),
                min_owned_apples=np.empty((num_episodes,), dtype=num_owned_apples.dtype),
                max_owned_apples=np.empty((num_episodes,), dtype=num_owned_apples.dtype),
                avg_owned_apples=np.empty((num_episodes,), dtype=np.float32),
                total_donated_apples=np.empty((num_episodes,), dtype=num_donated_apples.dtype),
                sum_reward=np.empty((num_episodes,), dtype=last_reward.dtype),
                episode_start=np.empty((num_episodes,), dtype=np.int),
            )
        )

    return np_stats


def build_summary(np_stats: Stats):
    for idx, agent_series in enumerate(np_stats.agent_series):
        # calculate per agent summary
        agent_series = np_stats.agent_series[idx]
        summary = agent_series.summary

        episode_start = 0
        for episode_idx, episode_end in enumerate(np_stats.episode_ends):
            episode_last_q = agent_series.last_q[episode_start:episode_end]
            episode_last_reward = agent_series.last_reward[episode_start:episode_end]
            num_owned_apples = agent_series.num_owned_apples[episode_start:episode_end]
            num_donated_apples = agent_series.num_donated_apples[episode_start:episode_end]
            num_taken_donations = agent_series.num_taken_donations[episode_start:episode_end]
            # todo: these groups of 3 could be merged into two single passes
            summary.max_q[episode_idx] = episode_last_q.max()
            summary.min_q[episode_idx] = episode_last_q.min()
            summary.avg_q[episode_idx] = episode_last_q.mean()
            summary.sum_reward[episode_idx] = episode_last_reward.sum()
            summary.min_owned_apples[episode_idx] = num_owned_apples.min()  # should actually always be zero
            summary.max_owned_apples[episode_idx] = num_owned_apples.max()
            summary.avg_owned_apples[episode_idx] = num_owned_apples.mean()
            summary.total_donated_apples[episode_idx] = num_donated_apples.sum() - num_taken_donations.sum()
            summary.episode_start[episode_idx] = episode_start
            episode_start = episode_end

    np_stats = np_stats._replace(
        summary=Summary(
            max_q=np_stats.agent_series[0].summary.max_q.copy(),
            min_q=np_stats.agent_series[0].summary.min_q.copy(),
            avg_q=np_stats.agent_series[0].summary.avg_q.copy(),
            sum_reward=np_stats.agent_series[0].summary.sum_reward.copy(),
            min_owned_apples=np_stats.agent_series[0].summary.min_owned_apples.astype(np.float32),
            max_owned_apples=np_stats.agent_series[0].summary.max_owned_apples.astype(np.float32),
            avg_owned_apples=np_stats.agent_series[0].summary.avg_owned_apples.astype(np.float32),
            total_donated_apples=np_stats.agent_series[0].summary.total_donated_apples.astype(np.float32),
            episode_start=np_stats.agent_series[0].summary.episode_start.copy(),
        )
    )

    summary = np_stats.summary
    num_not_nan = (~np.isnan(np_stats.summary.avg_q.data)).astype(np.int)
    for agent_series in np_stats.agent_series[1:]:
        np.add(num_not_nan, ~np.isnan(agent_series.summary.avg_q), out=num_not_nan)
        np.add(summary.max_q, agent_series.summary.max_q, out=summary.max_q)
        np.add(summary.min_q, agent_series.summary.min_q, out=summary.min_q)
        np.add(summary.avg_q, agent_series.summary.avg_q, out=summary.avg_q)
        np.add(summary.sum_reward, agent_series.summary.sum_reward, out=summary.sum_reward)
        np.add(summary.min_owned_apples, agent_series.summary.min_owned_apples, out=summary.min_owned_apples)
        np.add(summary.max_owned_apples, agent_series.summary.max_owned_apples, out=summary.max_owned_apples)
        np.add(summary.avg_owned_apples, agent_series.summary.avg_owned_apples, out=summary.avg_owned_apples)
        np.add(summary.total_donated_apples, agent_series.summary.total_donated_apples, out=summary.total_donated_apples)

    np.divide(summary.max_q, num_not_nan, out=summary.max_q, where=num_not_nan > 0)
    np.divide(summary.min_q, num_not_nan, out=summary.min_q, where=num_not_nan > 0)
    np.divide(summary.avg_q, num_not_nan, out=summary.avg_q, where=num_not_nan > 0)
    np.divide(summary.sum_reward, num_not_nan, out=summary.sum_reward, where=num_not_nan > 0)
    num_series = len(np_stats.agent_series)
    np.divide(summary.min_owned_apples, num_series, out=summary.sum_reward)
    np.divide(summary.max_owned_apples, num_series, out=summary.max_owned_apples)
    np.divide(summary.avg_owned_apples, num_series, out=summary.avg_owned_apples)
    np.divide(summary.total_donated_apples, num_series, out=summary.total_donated_apples)

    np.nan_to_num(summary.max_q, copy=False, nan=0.0)
    np.nan_to_num(summary.min_q, copy=False, nan=0.0)
    np.nan_to_num(summary.avg_q, copy=False, nan=0.0)
    np.nan_to_num(summary.sum_reward, copy=False, nan=0.0)

    return np_stats
