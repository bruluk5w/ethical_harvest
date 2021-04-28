import json
import os
from collections import namedtuple
from typing import Union
from threading import Lock

Config = namedtuple('Config', [
    # General config
    'EXPERIMENT_NAME',
    # Agent config
    'TARGET_NETWORK_UPDATE_FREQUENCY',
    'MINI_BATCH_SIZE'
    'DISCOUNT_FACTOR',
    'REPLAY_BUFFER_LENGTH',
    'REPLAY_BUFFER_MIN_LENGTH_FOR_USE',
    'EXPLORATION_ANNEALING',
    'EXPLORE_PROBABILITY_MIN',
    'EXPLORE_PROBABILITY_MAX',
])

__cfg_lock = Lock()
__cfg = None  # type: Union[None, Config]


def set_agent_config(**kwargs):
    global __cfg
    with __cfg_lock:
        if __cfg is None:
            __cfg = Config(**kwargs)
        else:
            __cfg = __cfg._replace(**kwargs)

        assert __cfg.REPLAY_BUFFER_LENGTH >= __cfg.REPLAY_BUFFER_MIN_LENGTH_FOR_USE


def save_cfg(filename='config.json'):
    path = os.path.join(get_storage(), filename)
    with __cfg_lock, open(path, 'w') as f:
        json.dump(__cfg, f)

    print('Saved config at \"{}\"'.format(path))


def load_cfg(filename='config.json'):
    global __cfg
    path = os.path.join(get_storage(), filename)
    with __cfg_lock, open(path, 'r') as f:
        __cfg = Config(*json.load(f))

    print('Loaded config from \"{}\"'.format(path))


def cfg():
    return __cfg


# General config
__DEFAULT_EXPERIMENT_NAME = 'default'
# Agent config
__DEFAULT_TARGET_NETWORK_UPDATE_FREQUENCY = 100
__DEFAULT_MINI_BATCH_SIZE = 32
__DEFAULT_DISCOUNT_FACTOR = 0.99
__DEFAULT_REPLAY_BUFFER_LENGTH = 1000  # 100000
__DEFAULT_REPLAY_BUFFER_MIN_LENGTH_FOR_USE = 10  # 10000
__DEFAULT_EXPLORATION_ANNEALING = 1 / 100000
__DEFAULT_EXPLORE_PROBABILITY_MIN = 0.05
# only start effective annealing when we begin using the replay buffer
__DEFAULT_EXPLORE_PROBABILITY_MAX = 1.0 + __DEFAULT_REPLAY_BUFFER_MIN_LENGTH_FOR_USE * __DEFAULT_EXPLORATION_ANNEALING

if __cfg is None:
    set_agent_config(
        # General config
        EXEPERIMENT_NAME=__DEFAULT_EXPERIMENT_NAME,
        # Agent config
        TARGET_NETWORK_UPDATE_FREQUENCY=__DEFAULT_TARGET_NETWORK_UPDATE_FREQUENCY,
        MINI_BATCH_SIZE=__DEFAULT_MINI_BATCH_SIZE,
        DISCOUNT_FACTOR=__DEFAULT_DISCOUNT_FACTOR,
        REPLAY_BUFFER_LENGTH=__DEFAULT_REPLAY_BUFFER_LENGTH,
        REPLAY_BUFFER_MIN_LENGTH_FOR_USE=__DEFAULT_REPLAY_BUFFER_MIN_LENGTH_FOR_USE,
        EXPLORATION_ANNEALING=__DEFAULT_EXPLORATION_ANNEALING,
        EXPLORE_PROBABILITY_MIN=__DEFAULT_EXPLORE_PROBABILITY_MIN,
        EXPLORE_PROBABILITY_MAX=__DEFAULT_EXPLORE_PROBABILITY_MAX,
    )


def get_storage():
    return os.path.join(os.getcwd(), cfg().EXPERIMENT_NAME)


def get_experiment_name():
    return os.listdir(os.getcwd())


def ensure_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
