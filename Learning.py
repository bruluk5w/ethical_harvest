from typing import List

import gym
import numpy as np
from gym.envs.registration import register

from constants import smallMap as map_sketch
from impl.agent import Agent
from impl.config import save_cfg
from envs.env import CommonsGame

register(
    id='CommonsGame-v0',
    entry_point='envs:CommonsGame',
)

"""
TODO: Decide the configuration parameters of the environment. We give some example ones.
"""

number_of_agents = 2

RENDER_ENV = False
SERVE_VISUALIZATION = True


def translate_state(state):
    return state


# def learning_policy(state, action_space, q_function):
#     """
#     TODO: Policy that the agent uses while it is training. As a template, here you have a completely random policy
#     :param state:  typically, a list of integers
#     :param action_space: a list of integers representing the possible actions that an agent can do
#     :param q_function: a numpy 2-d array (in tabular RL) that contains information about each state-action pair
#     :return: a recommended action
#     """
#
#     index = np.random.randint(low=0, high=len(action_space))
#     return action_space[index]


# def     update_q(q_function, current_state, current_action, current_reward, next_state):
#     """
#     TODO: method for updating the Q_function of the agent (this is the way it learns).
#     As a template, it updates Q(s,a) with the current reward
#
#     :param q_function: a numpy 2-d array (in tabular RL) that contains information about each state-action pair
#     :param current_state: a list of integers
#     :param current_action: an integer
#     :param current_reward: an integer
#     :param next_state: a list of integers
#     :return: an updated Q_function
#     """
#
#     q_function[current_state][current_action] = current_reward
#
#     return q_function

def new_action_space(action_space):
    action_space.remove(CommonsGame.SHOOT)
    action_space.remove(CommonsGame.TURN_CLOCKWISE)
    action_space.remove(CommonsGame.TURN_COUNTERCLOCKWISE)
    return action_space


def learning_loop(environment):
    """
    :param environment: the environment already configured
    :return:
    """

    action_space = np.array(new_action_space(list(range(environment.action_space.n))), dtype=np.int)

    # q_functions = np.zeros((number_of_agents, len_state_space, len(action_space)))

    agents = None  # type: List[Agent]
    last_rewards = None  # type: List[float]

    episodes = 10000
    timesteps = 1000

    episode_lengths = []
    for episode in range(episodes):
        new_states = [translate_state(observation) for observation in environment.reset()]  # type: List[np.ndarray]

        if agents is None:
            agents = [Agent(new_states[idx], action_space) for idx in range(number_of_agents)]
            last_rewards = [0.0] * number_of_agents

        for agent in agents:
            agent.new_episode()

        timestep = None
        for timestep in range(timesteps):
            print("--Episode", episode, ", Time step", timestep, "--")
            actions = []
            for agent, last_reward, new_state in zip(agents, last_rewards, new_states):
                # agent chooses index of action since action space is constant
                action_idx = agent.step(last_reward, new_state, training=True)
                actions.append(action_space[action_idx] if action_idx is not None else None)

            # for agent in agents:
            #     action_i = learning_policy(state[agent], action_space, q_functions[agent])
            #     n_actions.append(action_i)

            n_observations, n_rewards, n_done, n_info = environment.step(actions)
            new_states = [translate_state(observation) for observation in n_observations]

            if RENDER_ENV:
                environment.render()

            # for agent in range(number_of_agents):
            #     q_functions[agent] = update_q(q_functions[agent], state[agent], n_actions[agent],
            #     n_rewards[agent], next_state[agent])

            # TODO: wise to stop episode right when all apples are gone?
            if all(n_done):
                break

        episode_lengths.append(timestep)
        print(episode_lengths)

        for agent_idx, agent in enumerate(agents):
            if not(episode % 2):
                agent.save(agent_idx)

    return agents


if __name__ == '__main__':
    save_cfg()

    if SERVE_VISUALIZATION:
        from impl import vis
        vis.serve_visualization()

    tabular_rl = False
    env = gym.make('CommonsGame-v0', num_agents=number_of_agents, map_sketch=map_sketch, visual_radius=3,
                   full_state=False, tabular_state=tabular_rl)
    agents = learning_loop(env)
