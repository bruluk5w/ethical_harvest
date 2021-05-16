import atexit
from typing import List, Union

import gym
import numpy as np
from gym.envs.registration import register

from impl.agent import Agent
from impl.config import cfg, save_cfg
from impl.stats_to_file import QStatsDumper, get_trace_file_path, QStatsReader
import concurrent.futures

register(
    id='CommonsGame-v0',
    entry_point='envs:CommonsGame',
)

"""
TODO: Decide the configuration parameters of the environment. We give some example ones.
"""

RENDER_ENV = False
SERVE_VISUALIZATION = False


def translate_state(state):
    return state


def new_action_space(action_space):
    for removed_action in cfg().REMOVED_ACTIONS:
        action_space.remove(removed_action)

    return action_space


_threadPool = concurrent.futures.ThreadPoolExecutor()


def game_loop(environment, episodes=10000, timesteps=1000, train=True, episode_callback=None, frame_callback=None,
              start_episode=-1, verbose=True):
    """
    :param environment: the environment already configured
    :param episodes: the number of episodes to run
    :param timesteps: the number of frames per episode
    :param train: True if agents should learn, False if load a model from disk and just play
    :param episode_callback: if provided then it is called after the end of every episode with data gathered
    :param frame_callback: if provided then it is called with data from every frame
    :param start_episode: episode index to start from. If train is False, then this is the index used to build the file
    name for loading the agent's model from disk
    :param verbose If True then every frame some information may be printed to the console, else only at every episode
    end
    :return: The agents created
    """

    action_space = np.array(new_action_space(list(range(environment.action_space.n))), dtype=np.int)

    agents = None  # type: Union[None, List[Agent]]
    last_rewards = None  # type: Union[None, List[float]]

    episode_lengths = []

    for episode in range(start_episode, episodes):
        new_states = [translate_state(observation) for observation in environment.reset()]  # type: List[np.ndarray]

        if agents is None:
            agents = [Agent(new_states[idx], action_space, idx, episode) for idx in range(environment.num_agents)]
            if episode < 0:
                for agent in agents:
                    agent.save_model()

            last_rewards = [0.0] * environment.num_agents

        for agent in agents:
            agent.new_episode()

        timestep = None
        n_torture_frames = 0
        for timestep in range(timesteps):
            if verbose:
                print("--Episode", episode, ", Time step", timestep, "--")
            actions = [None] * len(agents)

            futures_to_agent = {_threadPool.submit(agent.step, last_reward, new_state, training=train): agent
                                for agent, last_reward, new_state in zip(agents, last_rewards, new_states)}

            for future in concurrent.futures.as_completed(futures_to_agent):
                agent = futures_to_agent[future]
                action_idx = future.result()
                actions[agent.agent_idx] = action_space[action_idx] if action_idx is not None else None

            n_observations, n_rewards, n_done, n_info = environment.step(actions)
            last_rewards = n_rewards  # rewards for the last action
            new_states = [translate_state(observation) for observation in n_observations]

            if RENDER_ENV:
                environment.render()

            if frame_callback:
                frame_callback(agents, episode)

            if all(n_done):
                n_torture_frames += 1
                if n_torture_frames > cfg().NUM_TORTURE_FRAMES:
                    break

        episode_lengths.append(timestep)
        print(episode_lengths)

        if train and not(episode % 2):
            for agent_idx, agent in enumerate(agents):
                agent.save_weights(agent_idx)

        if episode_callback is not None:
            episode_callback(agents, episode)

    return agents


def make_env():
    tabular_rl = False
    return gym.make('CommonsGame-v0', num_agents=cfg().NUM_AGENTS, map_sketch=cfg().MAP, visual_radius=3,
                    full_state=True, tabular_state=tabular_rl)


if __name__ == '__main__':
    save_cfg()

    if SERVE_VISUALIZATION:
        from impl import vis
        vis.serve_visualization()

    dumper = QStatsDumper(get_trace_file_path())
    reader = QStatsReader(get_trace_file_path(),)

    @atexit.register
    def cleanup():
        reader.stop()

    game_loop(
        make_env(),
        episode_callback=dumper.on_episode_end,
        frame_callback=dumper.on_episode_frame,
        verbose=True
    )
