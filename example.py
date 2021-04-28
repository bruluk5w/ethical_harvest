import numpy as np
import gym

from constants import tinyMap, smallMap, bigMap
from gym.envs.registration import register

register(
    id='CommonsGame-v0',
    entry_point='envs:CommonsGame',
)

numAgents = 2

env = gym.make('CommonsGame-v0', num_agents=numAgents, visual_radius=4, map_sketch=smallMap, full_state=True, tabular_state=True)
env.reset()

print(env.observation_space)

MOVE_UP = 0
MOVE_DOWN = 1
MOVE_LEFT = 2
MOVE_RIGHT = 3

TURN_CLOCKWISE = 4
TURN_COUNTERCLOCKWISE = 5
STAY = 6
SHOOT = 7
DONATE = 8
TAKE_DONATION = 9


for t in range(500):
    nActions = np.random.randint(low=0, high=env.action_space.n, size=(numAgents,)).tolist()
    nObservations, nRewards, nDone, nInfo = env.step(nActions)

    print("--Time step", t ,"--")
    env.render()

common_apples = 0
for n, agent in enumerate(env.get_agents()):
    print("Agent")
    print("Agent " + str(n) + " possessions : " + str(agent.has_apples))
    print("Agent " + str(n) + " donations : " + str(agent.donated_apples))
    print("Agent " + str(n) + "'s efficiency : " + str(agent.efficiency))
    common_apples += agent.donated_apples

    print("--")

print("Total common apples : ", common_apples)
