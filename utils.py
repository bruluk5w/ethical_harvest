import numpy as np
from pycolab.rendering import ObservationToArray


def build_map(map_sketch, num_pad_pixels, agent_chars):
    num_agents = len(agent_chars)
    game_map = np.array(map_sketch)

    def pad_with(vector, pad_width, iaxis, kwargs):
        del iaxis
        pad_value = kwargs.get('padder', ' ')
        vector[:pad_width[0]] = pad_value
        vector[-pad_width[1]:] = pad_value
        return vector

    # Put agents
    non_filled_spots = np.argwhere(np.logical_and(game_map != '@', game_map != '='))
    selected_spots = np.random.choice(non_filled_spots.shape[0], size=(num_agents,), replace=False)
    agents_coords = non_filled_spots[selected_spots, :]
    for idx, coord in enumerate(agents_coords):
        game_map[coord[0], coord[1]] = agent_chars[idx]
    # Put walls
    game_map = np.pad(game_map, num_pad_pixels + 1, pad_with, padder='=')

    game_map = [''.join(row.tolist()) for row in game_map]
    return game_map


class ObservationToArrayWithRGB(object):
    def __init__(self, colour_mapping):
        self._colour_mapping = colour_mapping
        # Rendering functions for the `board` representation and `RGB` values.
        self._renderers = {
            'RGB': ObservationToArray(value_mapping=colour_mapping)
        }

    def __call__(self, observation):
        # Perform observation rendering for agent and for video recording.
        result = {}
        for key, renderer in self._renderers.items():
            result[key] = renderer(observation)
        # Convert to [0, 255] RGB values.
        result['RGB'] = (result['RGB'] / 999.0 * 255.0).astype(np.uint8)
        return result
