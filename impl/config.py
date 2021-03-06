import json
import os
from collections import namedtuple
from threading import Lock
from typing import Union

from constants import MAPS
from envs import actions

Config = namedtuple('Config', [
    # General config
    'EXPERIMENT_NAME',
    # Agent training config
    'TARGET_NETWORK_UPDATE_FREQUENCY',
    'MINI_BATCH_SIZE',
    'DISCOUNT_FACTOR',
    'REPLAY_BUFFER_LENGTH',
    'REPLAY_BUFFER_MIN_LENGTH_FOR_USE',
    'EXPLORATION_ANNEALING',
    'EXPLORE_PROBABILITY_MIN',
    'EXPLORE_PROBABILITY_MAX',
    'NUM_TORTURE_FRAMES',
    # environment config
    'REMOVED_ACTIONS',
    'NUM_AGENTS',
    'MAP',
    'TOP_BAR_SHOWS_INEQUALITY',
    'USE_INEQUALITY_FOR_REWARD',
])

__cfg_lock = Lock()
__cfg = None  # type: Union[None, Config]


def set_config(**kwargs):
    global __cfg
    with __cfg_lock:
        if __cfg is None:
            __cfg = Config(**kwargs)
        else:
            __cfg = __cfg._replace(**kwargs)

        assert __cfg.REPLAY_BUFFER_LENGTH >= __cfg.REPLAY_BUFFER_MIN_LENGTH_FOR_USE


def save_cfg(filename='config.json'):
    path = os.path.join(get_storage(), filename)
    with __cfg_lock, open(path, 'w', encoding='utf-8') as f:
        json.dump(__cfg._asdict(), f)

    print('Saved config at \"{}\"'.format(path))


def load_cfg(filename='config.json'):
    global __cfg
    path = os.path.join(get_storage(), filename)
    with __cfg_lock, open(path, 'r', encoding='utf-8') as f:
        try:
            __cfg = Config(**json.load(f))
        except TypeError:
            print("Warning: config version not compatible. Trying to load with defaults for fields that are not present"
                  " in the config file!")
            new_cfg = __cfg._asdict()
            new_cfg.update(**json.load(f))
            __cfg = Config(**new_cfg)

    print('Loaded config from \"{}\"'.format(path))


def cfg():
    return __cfg


# General config
__DEFAULT_EXPERIMENT_NAME = 'default'
# Agent training config
__DEFAULT_TARGET_NETWORK_UPDATE_FREQUENCY = 1000
__DEFAULT_MINI_BATCH_SIZE = 32
__DEFAULT_DISCOUNT_FACTOR = 0.99
__DEFAULT_REPLAY_BUFFER_LENGTH = 10000  # 100000
__DEFAULT_REPLAY_BUFFER_MIN_LENGTH_FOR_USE = 100  # 10000
__DEFAULT_EXPLORATION_ANNEALING = 1 / 100000
__DEFAULT_EXPLORE_PROBABILITY_MIN = 0.01
__DEFAULT_NUM_TORTURE_FRAMES = 20
# only start effective annealing when we begin using the replay buffer
__DEFAULT_EXPLORE_PROBABILITY_MAX = 1.0 + __DEFAULT_REPLAY_BUFFER_MIN_LENGTH_FOR_USE * __DEFAULT_EXPLORATION_ANNEALING
# environment config
__DEFAULT_REMOVED_ACTIONS = [actions.TURN_CLOCKWISE, actions.TURN_COUNTERCLOCKWISE, actions.SHOOT]
__DEFAULT_NUM_AGENTS = 4
__DEFAULT_MAP = MAPS['smallMap']
__DEFAULT_TOP_BAR_SHOWS_INEQUALITY = False
__DEFAULT_USE_INEQUALITY_FOR_REWARD = False

if __cfg is None:
    set_config(
        # General config
        EXPERIMENT_NAME=__DEFAULT_EXPERIMENT_NAME,
        # Agent training config
        TARGET_NETWORK_UPDATE_FREQUENCY=__DEFAULT_TARGET_NETWORK_UPDATE_FREQUENCY,
        MINI_BATCH_SIZE=__DEFAULT_MINI_BATCH_SIZE,
        DISCOUNT_FACTOR=__DEFAULT_DISCOUNT_FACTOR,
        REPLAY_BUFFER_LENGTH=__DEFAULT_REPLAY_BUFFER_LENGTH,
        REPLAY_BUFFER_MIN_LENGTH_FOR_USE=__DEFAULT_REPLAY_BUFFER_MIN_LENGTH_FOR_USE,
        EXPLORATION_ANNEALING=__DEFAULT_EXPLORATION_ANNEALING,
        EXPLORE_PROBABILITY_MIN=__DEFAULT_EXPLORE_PROBABILITY_MIN,
        EXPLORE_PROBABILITY_MAX=__DEFAULT_EXPLORE_PROBABILITY_MAX,
        NUM_TORTURE_FRAMES=__DEFAULT_NUM_TORTURE_FRAMES,
        # environment config
        REMOVED_ACTIONS=__DEFAULT_REMOVED_ACTIONS,
        NUM_AGENTS=__DEFAULT_NUM_AGENTS,
        MAP=__DEFAULT_MAP,
        TOP_BAR_SHOWS_INEQUALITY=__DEFAULT_TOP_BAR_SHOWS_INEQUALITY,
        USE_INEQUALITY_FOR_REWARD=__DEFAULT_USE_INEQUALITY_FOR_REWARD,
    )


def get_storage():
    path = get_storage_for_experiment(cfg().EXPERIMENT_NAME)
    ensure_folder(path)
    return path


def get_storage_for_experiment(experiment_name):
    return os.path.join(os.getcwd(), experiment_name)


def get_experiment_names():
    return next(os.walk(os.getcwd()))[1]


def ensure_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
