import numpy as np
from pycolab.prefab_parts import sprites
from pycolab import things as pythings
from scipy.ndimage import convolve
from constants import *


class PlayerSprite(sprites.MazeWalker):
    def __init__(self, corner, position, character, agent_chars):
        super(PlayerSprite, self).__init__(
            corner, position, character, impassable=['='] + list(agent_chars.replace(character, '')),
            confined_to_board=True)
        self.agentChars = agent_chars
        self.orientation = np.random.choice(4)
        self.initPos = position
        self.visualRadius = 0
        self.timeout = 0

        # New variables
        self.has_apples = 0
        self.has_shot = False
        self.has_donated = False
        self.is_sick = False
        self.did_nothing = False
        self.took_donation = False
        self.efficiency = 1
        self.probability_getting_sick = 0
        self.donated_apples = 0

    def set_sickness(self, prob):
        if 0 <= prob <= 100:
            if AGENTS_CAN_GET_SICK:
                self.probability_getting_sick = prob

    def set_efficiency(self, prob):
        if 1 <= prob <= 6:
            if AGENTS_HAVE_DIFFERENT_EFFICIENCY:
                self.efficiency = prob

    def update(self, actions, board, layers, backdrop, things, the_plot):

        self.is_sick = self.probability_getting_sick > np.random.choice(100)
        if actions is not None:
            a = actions[self.agentChars.index(self.character)]
        else:
            return
        if self._visible:
            if things['.'].curtain[self.position[0], self.position[1]] or self.is_sick:
                self.timeout = TIMEOUT_FRAMES
                self._visible = False
            else:
                if a == 0:  # go upward?
                    if self.orientation == 0:
                        self._north(board, the_plot)
                    elif self.orientation == 1:
                        self._east(board, the_plot)
                    elif self.orientation == 2:
                        self._south(board, the_plot)
                    elif self.orientation == 3:
                        self._west(board, the_plot)
                elif a == 1:  # go downward?
                    if self.orientation == 0:
                        self._south(board, the_plot)
                    elif self.orientation == 1:
                        self._west(board, the_plot)
                    elif self.orientation == 2:
                        self._north(board, the_plot)
                    elif self.orientation == 3:
                        self._east(board, the_plot)
                elif a == 2:  # go leftward?
                    if self.orientation == 0:
                        self._west(board, the_plot)
                    elif self.orientation == 1:
                        self._north(board, the_plot)
                    elif self.orientation == 2:
                        self._east(board, the_plot)
                    elif self.orientation == 3:
                        self._south(board, the_plot)
                elif a == 3:  # go rightward?
                    if self.orientation == 0:
                        self._east(board, the_plot)
                    elif self.orientation == 1:
                        self._south(board, the_plot)
                    elif self.orientation == 2:
                        self._west(board, the_plot)
                    elif self.orientation == 3:
                        self._north(board, the_plot)
                elif a == 4:  # turn right?
                    if self.orientation == 3:
                        self.orientation = 0
                    else:
                        self.orientation = self.orientation + 1
                elif a == 5:  # turn left?
                    if self.orientation == 0:
                        self.orientation = 3
                    else:
                        self.orientation = self.orientation - 1
                elif a == 6:  # do nothing?
                    self.did_nothing = True
                    self._stay(board, the_plot)
                elif a == 8:  # donate?
                    if self.has_apples > 0:
                        self.has_donated = True
                    self._stay(board, the_plot)
                elif a == 9:  # took donation?
                    self.took_donation = True
                    self._stay(board, the_plot)
        else:
            if self.timeout == 0:
                self._teleport(self.initPos)
                self._visible = True
            else:
                self.timeout -= 1


class SightDrape(pythings.Drape):
    """Scope of agent Drape"""

    def __init__(self, curtain, character, agent_chars, num_pad_pixels):
        super().__init__(curtain, character)
        self.agentChars = agent_chars
        self.numPadPixels = num_pad_pixels
        self.h = curtain.shape[0] - (num_pad_pixels * 2 + 2)
        self.w = curtain.shape[1] - (num_pad_pixels * 2 + 2)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        np.logical_and(self.curtain, False, self.curtain)
        ags = [things[c] for c in self.agentChars]
        for agent in ags:
            if agent.visible:
                pos = agent.position
                if agent.orientation == 0:
                    self.curtain[pos[0] - 1, pos[1]] = True
                elif agent.orientation == 1:
                    self.curtain[pos[0], pos[1] + 1] = True
                elif agent.orientation == 2:
                    self.curtain[pos[0] + 1, pos[1]] = True
                elif agent.orientation == 3:
                    self.curtain[pos[0], pos[1] - 1] = True
                self.curtain[:, :] = np.logical_and(self.curtain, np.logical_not(layers['=']))


class ShotDrape(pythings.Drape):
    """Tagging ray Drape"""

    def __init__(self, curtain, character, agent_chars, num_pad_pixels):
        super().__init__(curtain, character)
        self.agentChars = agent_chars
        self.numPadPixels = num_pad_pixels
        self.h = curtain.shape[0] - (num_pad_pixels * 2 + 2)
        self.w = curtain.shape[1] - (num_pad_pixels * 2 + 2)
        self.scopeHeight = num_pad_pixels + 1

    def update(self, actions, board, layers, backdrop, things, the_plot):
        beam_width = 0
        beam_height = self.scopeHeight
        np.logical_and(self.curtain, False, self.curtain)
        if actions is not None:
            for i, a in enumerate(actions):
                if a == 7:
                    things[self.agentChars[i]].has_shot = True
                    agent = things[self.agentChars[i]]
                    if agent.visible:
                        pos = agent.position
                        if agent.orientation == 0:
                            if np.any(layers['='][pos[0] - beam_height:pos[0],
                                      pos[1] - beam_width:pos[1] + beam_width + 1]):
                                collision_idxs = np.argwhere(layers['='][pos[0] - beam_height:pos[0],
                                                             pos[1] - beam_width:pos[1] + beam_width + 1])
                                beam_height = beam_height - (np.max(collision_idxs) + 1)
                            self.curtain[pos[0] - beam_height:pos[0],
                                         pos[1] - beam_width:pos[1] + beam_width + 1] = True
                        elif agent.orientation == 1:
                            if np.any(layers['='][pos[0] - beam_width:pos[0] + beam_width + 1,
                                      pos[1] + 1:pos[1] + beam_height + 1]):
                                collision_idxs = np.argwhere(layers['='][pos[0] - beam_width:pos[0] + beam_width + 1,
                                                             pos[1] + 1:pos[1] + beam_height + 1])
                                beam_height = np.min(collision_idxs)
                            self.curtain[pos[0] - beam_width:pos[0] + beam_width + 1,
                                         pos[1] + 1:pos[1] + beam_height + 1] = True
                        elif agent.orientation == 2:
                            if np.any(layers['='][pos[0] + 1:pos[0] + beam_height + 1,
                                      pos[1] - beam_width:pos[1] + beam_width + 1]):
                                collision_idxs = np.argwhere(layers['='][pos[0] + 1:pos[0] + beam_height + 1,
                                                             pos[1] - beam_width:pos[1] + beam_width + 1])
                                beam_height = np.min(collision_idxs)
                            self.curtain[pos[0] + 1:pos[0] + beam_height + 1,
                                         pos[1] - beam_width:pos[1] + beam_width + 1] = True
                        elif agent.orientation == 3:
                            if np.any(layers['='][pos[0] - beam_width:pos[0] + beam_width + 1,
                                      pos[1] - beam_height:pos[1]]):
                                collision_idxs = np.argwhere(layers['='][pos[0] - beam_width:pos[0] + beam_width + 1,
                                                             pos[1] - beam_height:pos[1]])
                                beam_height = beam_height - (np.max(collision_idxs) + 1)
                            self.curtain[pos[0] - beam_width:pos[0] + beam_width + 1,
                                         pos[1] - beam_height:pos[1]] = True
                        # self.curtain[:, :] = np.logical_and(self.curtain, np.logical_not(layers['=']))
        else:
            return


class AppleDrape(pythings.Drape):
    """Coins Drape"""

    def __init__(self, curtain, character, agent_chars, num_pad_pixels):
        super().__init__(curtain, character)
        self.agentChars = agent_chars
        self.numPadPixels = num_pad_pixels
        self.apples = np.copy(curtain)
        self.common_pool = 0

    def update(self, actions, board, layers, backdrop, things, the_plot):
        rewards = []

        agents_map = np.ones(self.curtain.shape, dtype=bool)
        for i in range(len(self.agentChars)):
            agent_efficiency = things[self.agentChars[i]].efficiency  # The number of apples it can collect on each turn
            rew = self.curtain[things[self.agentChars[i]].position[0], things[self.agentChars[i]].position[1]]
            greedy = False  # A greedy agent takes more apples than what it needs
            not_stupid = False  # A stupid agent does not take more apples when it needs them
            if rew:
                self.curtain[things[self.agentChars[i]].position[0], things[self.agentChars[i]].position[1]] = False
                things[self.agentChars[i]].has_apples += agent_efficiency
                greedy = things[self.agentChars[i]].has_apples > TOO_MANY_APPLES

            elif things[self.agentChars[i]].did_nothing:
                not_stupid = things[self.agentChars[i]].has_apples > TOO_MANY_APPLES
                things[self.agentChars[i]].did_nothing = False
            else:
                things[self.agentChars[i]].did_nothing = False

            donation = things[self.agentChars[i]].has_donated
            took_donation = things[self.agentChars[i]].took_donation
            shot = things[self.agentChars[i]].has_shot
            if donation:
                things[self.agentChars[i]].has_donated = False
                things[self.agentChars[i]].has_apples -= 1
                things[self.agentChars[i]].donated_apples += 1
                self.common_pool += 1
            elif took_donation:
                things[self.agentChars[i]].took_donation = False
                if self.common_pool > 0:
                    self.common_pool -= 1
                    things[self.agentChars[i]].has_apples += 1
                    greedy = things[self.agentChars[i]].has_apples > TOO_MANY_APPLES
            elif shot:
                things[self.agentChars[i]].has_shot = False

            if things[self.agentChars[i]].timeout > 0:
                rewards.append(0)
            else:
                # The rewards takes into account if an apple has been gathered or if an apple has been donated
                rewards.append(
                    rew * APPLE_GATHERING_REWARD +
                    greedy * TOO_MANY_APPLES_PUNISHMENT +
                    not_stupid * DID_NOTHING_BECAUSE_MANY_APPLES_REWARD +
                    donation * DONATION_REWARD +
                    took_donation * TOOK_DONATION_REWARD +
                    shot * SHOOTING_PUNISHMENT
                )

            agents_map[things[self.agentChars[i]].position[0], things[self.agentChars[i]].position[1]] = False

        the_plot.add_reward(rewards)
        # Matrix of local stock of apples
        kernel = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 0, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
        num_local_apples = convolve(self.curtain[self.numPadPixels + 1:-self.numPadPixels - 1,
                                    self.numPadPixels + 1:-self.numPadPixels - 1] * 1, kernel, mode='constant')
        probs = np.zeros(num_local_apples.shape)

        probs[(num_local_apples > 0) & (num_local_apples <= 2)] = RESPAWN_PROBABILITIES[0]
        probs[(num_local_apples > 2) & (num_local_apples <= 4)] = RESPAWN_PROBABILITIES[1]
        probs[(num_local_apples > 4)] = RESPAWN_PROBABILITIES[2]

        ags = [things[c] for c in self.agentChars]
        num_agent = 0

        x_agent = self.numPadPixels + 1
        y_agent = self.numPadPixels + 1

        for agent in ags:
            if agent.has_apples > 1:
                self.apples[x_agent, y_agent + 3 * num_agent] = True
                self.curtain[x_agent, y_agent + 3 * num_agent] = True

                self.apples[x_agent, y_agent + 1 + 3 * num_agent] = True
                self.curtain[x_agent, y_agent + 1 + 3 * num_agent] = True
            elif agent.has_apples > 0:
                self.apples[x_agent, y_agent + 3 * num_agent] = True
                self.curtain[x_agent, y_agent + 3 * num_agent] = True

                self.apples[x_agent, y_agent + 1 + 3 * num_agent] = False
                self.curtain[x_agent, y_agent + 1 + 3 * num_agent] = False
            else:
                self.apples[x_agent, y_agent + 3 * num_agent] = False
                self.curtain[x_agent, y_agent + 3 * num_agent] = False

                self.apples[x_agent, y_agent + 1 + 3 * num_agent] = False
                self.curtain[x_agent, y_agent + 1 + 3 * num_agent] = False
            num_agent += 1

        apple_idxs = np.argwhere(np.logical_and(np.logical_xor(self.apples, self.curtain), agents_map))

        for i, j in apple_idxs:
            if SUSTAINABILITY_MATTERS:
                self.curtain[i, j] = np.random.choice([True, False],
                                                      p=[probs[i - self.numPadPixels - 1, j - self.numPadPixels - 1],
                                                         1 - probs[
                                                             i - self.numPadPixels - 1, j - self.numPadPixels - 1]])
            else:
                self.curtain[i, j] = np.random.choice([True, False],
                                                      p=[REGENERATION_PROBABILITY, 1 - REGENERATION_PROBABILITY])
