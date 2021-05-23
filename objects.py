import math

import numpy as np
from pycolab import things as pythings
from pycolab.prefab_parts import sprites
from scipy.ndimage import convolve

from constants import *
from envs import actions as action_codes
from impl.config import cfg


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
        self.taken_apples = 0

    def get_agent_config(self):
        return {
            'efficiency': int(self.efficiency),
            'probability_getting_sick': int(self.probability_getting_sick),
        }

    def set_agent_config(self, config):
        self.efficiency = config['efficiency']
        self.probability_getting_sick = config['probability_getting_sick']

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
                if a == action_codes.MOVE_UP:
                    if self.orientation == 0:
                        self._north(board, the_plot)
                    elif self.orientation == 1:
                        self._east(board, the_plot)
                    elif self.orientation == 2:
                        self._south(board, the_plot)
                    elif self.orientation == 3:
                        self._west(board, the_plot)
                elif a == action_codes.MOVE_DOWN:
                    if self.orientation == 0:
                        self._south(board, the_plot)
                    elif self.orientation == 1:
                        self._west(board, the_plot)
                    elif self.orientation == 2:
                        self._north(board, the_plot)
                    elif self.orientation == 3:
                        self._east(board, the_plot)
                elif a == action_codes.MOVE_LEFT:
                    if self.orientation == 0:
                        self._west(board, the_plot)
                    elif self.orientation == 1:
                        self._north(board, the_plot)
                    elif self.orientation == 2:
                        self._east(board, the_plot)
                    elif self.orientation == 3:
                        self._south(board, the_plot)
                elif a == action_codes.MOVE_RIGHT:
                    if self.orientation == 0:
                        self._east(board, the_plot)
                    elif self.orientation == 1:
                        self._south(board, the_plot)
                    elif self.orientation == 2:
                        self._west(board, the_plot)
                    elif self.orientation == 3:
                        self._north(board, the_plot)
                elif a == action_codes.TURN_CLOCKWISE:
                    if self.orientation == 3:
                        self.orientation = 0
                    else:
                        self.orientation = self.orientation + 1
                elif a == action_codes.TURN_COUNTERCLOCKWISE:
                    if self.orientation == 0:
                        self.orientation = 3
                    else:
                        self.orientation = self.orientation - 1
                elif a == action_codes.STAY:
                    self.did_nothing = True
                    self._stay(board, the_plot)
                elif a == action_codes.DONATE:
                    if self.has_apples > 0:
                        self.has_donated = True
                    self._stay(board, the_plot)
                elif a == action_codes.TAKE_DONATION:
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
        ags = [things[c] for c in self.agentChars]
        agents_map = np.ones(self.curtain.shape, dtype=bool)
        reward_args = []
        for ag in ags:
            args = {}
            reward_args.append(args)
            agent_efficiency = ag.efficiency  # The number of apples it can collect on each turn
            rew = self.curtain[ag.position[0], ag.position[1]]
            greedy = False  # A greedy agent takes more apples than what it needs
            not_stupid = False  # A stupid agent does not take more apples when it needs them
            if rew:
                self.curtain[ag.position[0], ag.position[1]] = False
                ag.has_apples += agent_efficiency
                greedy = ag.has_apples > TOO_MANY_APPLES

            elif ag.did_nothing:
                not_stupid = ag.has_apples > TOO_MANY_APPLES
                ag.did_nothing = False
            else:
                ag.did_nothing = False

            donation = ag.has_donated
            took_donation = ag.took_donation
            shot = ag.has_shot
            ag.donated_apples = 0
            ag.taken_apples = 0
            if donation:
                ag.has_donated = False
                ag.has_apples -= 1
                ag.donated_apples = 1
                self.common_pool += 1
            elif took_donation:
                ag.took_donation = False
                if self.common_pool > 0:
                    self.common_pool -= 1
                    ag.has_apples += 1
                    ag.taken_apples = 1
                    greedy = ag.has_apples > TOO_MANY_APPLES
            elif shot:
                ag.has_shot = False

            args['rew'] = rew
            args['greedy'] = greedy
            args['not_stupid'] = not_stupid
            args['donation'] = donation
            args['took_donation'] = took_donation
            args['shot'] = shot

        target_apples = math.sqrt(sum(ag.has_apples * ag.has_apples for ag in ags) / len(ags))  # using sum of squared values to encourage better performance for everyone

        for ag, args in zip(ags, reward_args):
            if ag.timeout > 0:
                rewards.append(0)
            else:
                if cfg().USE_INEQUALITY_FOR_REWARD:
                    overperformance = max(0, ag.has_apples - target_apples) # overperformance is how many apples too much
                    underperformance = max(0,  target_apples -ag.has_apples) # overperformance is how many apples too much
                    donation_reward = -DONATION_REWARD if overperformance > 0 else DONATION_REWARD  # no punishment for donation if overperforming
                    overperformance_reward = 0.1 if args['donation'] else -overperformance * 0.2  # general penalty if overperforming
                    if overperformance_reward < 0:
                        overperformance_reward = -math.log2(-overperformance_reward + 1) / 2.0
                    rewards.append(
                        args['rew'] * APPLE_GATHERING_REWARD +
                        args['greedy'] * TOO_MANY_APPLES_PUNISHMENT * (overperformance > 0) +  # no greedy punishment when underperforming (should allow everyone to have more apples than the minimum required if a lot are available)
                        args['not_stupid'] * DID_NOTHING_BECAUSE_MANY_APPLES_REWARD +
                        args['donation'] * donation_reward +
                        args['took_donation'] * TOOK_DONATION_REWARD * (1 + 0.2 * underperformance) +
                        args['shot'] * SHOOTING_PUNISHMENT +
                        overperformance_reward
                    )
                else:
                    # The rewards takes into account if an apple has been gathered or if an apple has been donated
                    rewards.append(
                        args['rew'] * APPLE_GATHERING_REWARD +
                        args['greedy'] * TOO_MANY_APPLES_PUNISHMENT +
                        args['not_stupid'] * DID_NOTHING_BECAUSE_MANY_APPLES_REWARD +
                        args['donation'] * DONATION_REWARD +
                        args['took_donation'] * TOOK_DONATION_REWARD +
                        args['shot'] * SHOOTING_PUNISHMENT
                    )

            agents_map[ag.position[0], ag.position[1]] = False

        the_plot.add_reward(rewards)

        # calculate apple respawn probability
        # Matrix of local stock of apples
        kernel = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 0, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
        apple_sources = self.curtain[self.numPadPixels + 1:-self.numPadPixels - 1,
                                     self.numPadPixels + 1:-self.numPadPixels - 1] * 1
        apple_sources[0] = 0  # delete top bar of apples display
        num_local_apples = convolve(apple_sources, kernel, mode='constant')
        probs = np.zeros(num_local_apples.shape)

        probs[(num_local_apples > 0) & (num_local_apples <= 2)] = RESPAWN_PROBABILITIES[0]
        probs[(num_local_apples > 2) & (num_local_apples <= 4)] = RESPAWN_PROBABILITIES[1]
        probs[(num_local_apples > 4)] = RESPAWN_PROBABILITIES[2]

        # draw apple display in top bar
        agent_idx = 0
        x_offset = self.numPadPixels + 1
        y_offset = self.numPadPixels + 1
        if cfg().TOP_BAR_SHOWS_INEQUALITY:
            # show underperformance of agents in top bar
            for agent in ags:
                underperformance = max(0, target_apples - agent.has_apples)
                if underperformance > 1.5:
                    self.apples[x_offset, y_offset + 3 * agent_idx] = True
                    self.curtain[x_offset, y_offset + 3 * agent_idx] = True

                    self.apples[x_offset, y_offset + 1 + 3 * agent_idx] = True
                    self.curtain[x_offset, y_offset + 1 + 3 * agent_idx] = True
                if underperformance > 3.5:
                    self.apples[x_offset, y_offset + 3 * agent_idx] = True
                    self.curtain[x_offset, y_offset + 3 * agent_idx] = True

                    self.apples[x_offset, y_offset + 1 + 3 * agent_idx] = False
                    self.curtain[x_offset, y_offset + 1 + 3 * agent_idx] = False
                else:
                    self.apples[x_offset, y_offset + 3 * agent_idx] = False
                    self.curtain[x_offset, y_offset + 3 * agent_idx] = False

                    self.apples[x_offset, y_offset + 1 + 3 * agent_idx] = False
                    self.curtain[x_offset, y_offset + 1 + 3 * agent_idx] = False

                agent_idx += 1

        else:
            for agent in ags:
                if agent.has_apples > 1:
                    self.apples[x_offset, y_offset + 3 * agent_idx] = True
                    self.curtain[x_offset, y_offset + 3 * agent_idx] = True

                    self.apples[x_offset, y_offset + 1 + 3 * agent_idx] = True
                    self.curtain[x_offset, y_offset + 1 + 3 * agent_idx] = True
                elif agent.has_apples > 0:
                    self.apples[x_offset, y_offset + 3 * agent_idx] = True
                    self.curtain[x_offset, y_offset + 3 * agent_idx] = True

                    self.apples[x_offset, y_offset + 1 + 3 * agent_idx] = False
                    self.curtain[x_offset, y_offset + 1 + 3 * agent_idx] = False
                else:
                    self.apples[x_offset, y_offset + 3 * agent_idx] = False
                    self.curtain[x_offset, y_offset + 3 * agent_idx] = False

                    self.apples[x_offset, y_offset + 1 + 3 * agent_idx] = False
                    self.curtain[x_offset, y_offset + 1 + 3 * agent_idx] = False

                agent_idx += 1

        apple_idxs = np.argwhere(np.logical_and(np.logical_xor(self.apples, self.curtain), agents_map))

        # respawn apples
        for i, j in apple_idxs:
            if SUSTAINABILITY_MATTERS:
                self.curtain[i, j] = np.random.choice([True, False],
                                                      p=[probs[i - self.numPadPixels - 1, j - self.numPadPixels - 1],
                                                         1 - probs[
                                                             i - self.numPadPixels - 1, j - self.numPadPixels - 1]])
            else:
                self.curtain[i, j] = np.random.choice([True, False],
                                                      p=[REGENERATION_PROBABILITY, 1 - REGENERATION_PROBABILITY])
