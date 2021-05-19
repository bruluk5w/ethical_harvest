import random

import cv2
import numpy as np
import gym
from gym import spaces
from pycolab import ascii_art
from utils import build_map, ObservationToArrayWithRGB
from objects import PlayerSprite, AppleDrape, SightDrape, ShotDrape
from constants import TIMEOUT_FRAMES


class CommonsGame(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

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

    def __init__(self, num_agents, visual_radius, map_sketch, full_state, tabular_state):
        super(CommonsGame, self).__init__()
        self.full_state = full_state
        # Setup spaces
        self.action_space = spaces.Discrete(10)
        ob_height = ob_width = visual_radius * 2 + 1
        # Setup game
        self.num_agents = num_agents
        self.sightRadius = visual_radius
        self.agentChars = agentChars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"[0:num_agents]
        self.mapHeight = len(map_sketch)
        self.mapWidth = len(map_sketch[0])
        self.tabularState = tabular_state
        self.common_pool = True

        if tabular_state:
            full_state = True
            self.full_state = True

        if full_state:
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.mapHeight + 2, self.mapWidth + 2, 3),
                                                dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=(ob_height, ob_width, 3), dtype=np.uint8)
        self.numPadPixels = numPadPixels = visual_radius - 1
        self.gameField = build_map(map_sketch, num_pad_pixels=numPadPixels, agent_chars=agentChars)

        self.state = None
        self.sick_probabilities = np.random.choice(100, num_agents)
        self.efficiency_probabilities = np.random.randint(1, 5, num_agents)

        # Pycolab related setup:
        self._game = self.build_game()
        colour_map = dict([(a, (999, 0, 0)) for i, a in enumerate(agentChars)]  # Agents
                          + [('=', (705, 705, 705))]  # Steel Impassable wall
                          + [(' ', (0, 0, 0))]  # Black background
                          + [('@', (0, 999, 0))]  # Green Apples
                          + [('.', (750, 750, 0))]  # Yellow beam
                          + [('-', (200, 200, 200))])  # Grey scope
        self.obToImage = ObservationToArrayWithRGB(colour_mapping=colour_map)

    def build_game(self):
        agents_order = list(self.agentChars)
        random.shuffle(agents_order)
        return ascii_art.ascii_art_to_game(
            self.gameField,
            what_lies_beneath=' ',
            sprites=dict(
                [(a, ascii_art.Partial(PlayerSprite, self.agentChars)) for a in self.agentChars]),
            drapes={'@': ascii_art.Partial(AppleDrape, self.agentChars, self.numPadPixels),
                    '-': ascii_art.Partial(SightDrape, self.agentChars, self.numPadPixels),
                    '.': ascii_art.Partial(ShotDrape, self.agentChars, self.numPadPixels)},
            # update_schedule=['.'] + agents_order + ['-'] + ['@'],
            update_schedule=['.'] + agents_order + ['-'] + ['@'],
            z_order=['-'] + ['@'] + agents_order + ['.']
        )

    def step(self, n_actions):
        n_info = {'n': []}
        self.state, n_rewards, _ = self._game.play(n_actions)

        n_observations, done = self.get_observation()
        n_done = [done] * self.num_agents
        return n_observations, n_rewards, n_done, n_info

    def reset(self):
        # Reset the state of the environment to an initial state
        self._game = self.build_game()
        ags = [self._game.things[c] for c in self.agentChars]
        for i, a in enumerate(ags):
            a.set_sickness(self.sick_probabilities[i])
            a.set_efficiency(self.efficiency_probabilities[i])

        self.state, _, _ = self._game.its_showtime()
        n_observations, _ = self.get_observation()
        return n_observations

    def render(self, mode='human', close=False, frame_idx=-1):
        # Render the environment to the screen
        board = self.obToImage(self.state)['RGB'].transpose([1, 2, 0])
        board = board[self.numPadPixels:self.numPadPixels + self.mapHeight + 2,
                self.numPadPixels:self.numPadPixels + self.mapWidth + 2, :]

        board = cv2.resize(board, (500, 500), interpolation=cv2.INTER_NEAREST)
        ags = [self._game.things[c] for c in self.agentChars]
        plot_text = "Frame {}\n".format(frame_idx)
        for i, agent in enumerate(ags):
            plot_text += "Agent " + str(i) + ": " + str(agent.has_apples) + ", "
        plot_text += "Common: " + str(self._game.things['@'].common_pool)
        cv2.putText(board, plot_text, (40, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=(255, 255, 0), thickness=1)

        cv2.imshow('Environment', cv2.cvtColor(board, cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)

    def get_observation(self):
        done = not (np.logical_or.reduce(self.state.layers['@'][5:, :], axis=None))
        ags = [self._game.things[c] for c in self.agentChars]
        obs = []

        new_state = self.state.board[self.sightRadius + 2:-self.sightRadius, self.sightRadius:-self.sightRadius]
        common_apples = self._game.things['@'].common_pool
        board = self.obToImage(self.state)['RGB'].transpose([1, 2, 0])
        num_agents = 0
        for a in ags:
            if a.visible or a.timeout == TIMEOUT_FRAMES:
                if self.full_state:

                    ob = np.copy(board)
                    if a.visible:
                        ob[a.position[0], a.position[1], :] = [0, 0, 255]
                    ob = ob[self.numPadPixels:self.numPadPixels + self.mapHeight + 2,
                            self.numPadPixels:self.numPadPixels + self.mapWidth + 2, :]

                else:
                    ob_apples = np.copy(board[4, 4 + 3*num_agents:5 + 1 + 3*num_agents, :])
                    relleno = np.copy(board[0, : 2 * self.sightRadius - 1, :])

                    ob = np.copy(board[
                                 a.position[0] - self.sightRadius:a.position[0] + self.sightRadius + 1,
                                 a.position[1] - self.sightRadius:a.position[1] + self.sightRadius + 1, :])

                    ob_apples = np.vstack((ob_apples, relleno))
                    ob = np.concatenate((ob, np.array([ob_apples])))
                    # print(ob)
                    if a.visible:
                        ob[self.sightRadius, self.sightRadius, :] = [0, 0, 255]
                ob = ob / 255.0
            else:
                ob = None
            new_state = np.append(new_state, [a.position[0] - self.sightRadius - 2,
                                              a.position[1] - self.sightRadius,
                                              a.has_apples, a.donated_apples])

            if not self.tabularState:
                obs.append(ob)
            num_agents += 1
        new_state = np.append(new_state, [common_apples])
        # print("State : ", new_state)

        if self.tabularState:
            for a in ags:
                if a.visible or a.timeout == TIMEOUT_FRAMES:
                    obs.append(new_state)
                else:
                    obs.append([])
        return obs, done

    def get_agents(self):
        return [self._game.things[c] for c in self.agentChars]

    def get_apple_drape(self):
        return self._game.things['@']
