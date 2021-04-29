import os
from itertools import groupby, islice
from typing import Union, Dict, Tuple, List

from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Slider, Dropdown, CustomJSTransform
from bokeh.plotting import figure
from bokeh.server.server import Server
from threading import Thread, Lock

from bokeh.transform import transform

from impl.config import get_experiment_names, get_storage, cfg, load_cfg, set_config
from impl.js_transformers import shift_right, shift_left, mean
from copy import deepcopy
from constants import get_model_name_params


class VisApp:

    def __init__(self, doc):
        if _STANDALONE:
            experiment_names = get_experiment_names()
            self._name = experiment_names[0] if experiment_names else None  # name of the selected experiment
            self._experiment_dropdown = Dropdown(label="No experiments available" if self._name is None else self._name,
                                                 menu=experiment_names)
        else:
            self._name = cfg().EXPERIMENT_NAME
            assert self._name is not None
            self._experiment_dropdown = Dropdown(label=self._name)

        self._storage = None  # folder for all the data of the experiment
        self._experiment_dropdown.on_click(self._on_experiment_change)
        self._q_series = {'episode': [], 'q_max': [], 'q_min': [], 'q_avg': [], 'real_q': []}
        self._q_series_src = ColumnDataSource(self._q_series)

        self._value_estimates_plot = figure(x_axis_label='episodes', y_axis_label='value_estimates',
                                            title="Size of q values")

        self._value_estimates_plot.quad(top='q_max', bottom='q_min',
                                        left=transform('episode', shift_left), right=transform('episode', shift_right),
                                        source=self._q_series_src,
                                        color='red', alpha=0.5)
        self._value_estimates_plot.line(x='episode', y='q_avg', source=self._q_series_src, color='red')
        self._value_estimates_plot.line(x='episode', y=transform('real_q', mean),
                                        source=self._q_series_src,
                                        color='blue')

        self._load_experiment(self._name)

        # def callback(attr, old, new):
        #     if new == 0:
        #         data = df
        #     else:
        #         data = df.rolling(f"{new}D").mean()
        #     source.data = ColumnDataSource.from_df(data)
        #
        # slider = Slider(start=0, end=30, value=0, step=1, title="Smoothing by N Days")
        # slider.on_change('value', callback)

        doc.add_root(column(self._experiment_dropdown, self._value_estimates_plot))

    def _on_experiment_change(self, event):
        self._experiment_dropdown.label = event.item
        self._load_experiment(event.item)

    def _load_experiment(self, name, ):
        global _STANDALONE

        if name is None:
            return

        if not _STANDALONE:
            return
        else:
            set_config(EXPERIMENT_NAME=name)
            backup = deepcopy(cfg())
            try:
                load_cfg()
            except FileNotFoundError:
                print("Error: config file is missing for the selected experiment, some things may not work as expected")
            except TypeError:
                print("Error: config file incompatible, some things may not work as expected")
                set_config(**backup._asdict())

            self._storage = get_storage()

        for seq in self._q_series.values():
            seq.clear()

        individual_model_params = (get_model_name_params(name) for name in
                                   next(os.walk(os.path.join(self._storage, 'models')))[1])
        episode_idx_key = lambda params: params[0]
        individual_model_params = sorted((params for params in individual_model_params if params[0] is not None),
                                         key=episode_idx_key)
        frames = ((key, *islice(zip(*group), 1, None)) for key, group in groupby(individual_model_params, key=episode_idx_key))

        for episode_idx, agent_indices, model_types in frames:
            #model_path = os.path.join(self._storage, saved_model)
            #model = load_model(model_path, episode, )
            self._load_episode_frame(episode_idx, agent_indices, model_types)

    def _load_episode_frame(self, episode_idx: int, agent_indices: List[int], model_types: List[str]):
        agent_indices = sorted(k for k, _ in groupby(agent_indices))
        if agent_indices != list(range(len(agent_indices))) or len(agent_indices) != cfg().NUM_AGENTS:
            print("Saved models for episode {} missing. Skipping frame.".format(episode_idx))
            return

        frames = 0
        rewards = 0
        max_q = float('-inf')
        min_q = float('+inf')
        sum_q = 0
        # episode = -1

        def accumulated_q_stats(agents, episode_idx, episode_length):
            nonlocal frames, rewards, max_q, min_q, sum_q  # , episode
            # episode = episode_idx
            for agent in agents:
                # read the predictions of the agent on the expected total discounted rewards for the actions
                q = max(agent.last_online_q_values)
                max_q, min_q = max(max_q, q), min(min_q, q)
                sum_q += q
                rewards += agent.sum_rewards
                frames += episode_length

        game_loop(make_env(), episodes=1, train=False, episode_callback=accumulated_q_stats, start_episode=episode_idx)

        self._q_series['episode'].append(episode_idx)
        self._q_series['max_q'].append(max_q)
        self._q_series['min_q'].append(min_q)
        self._q_series['avg_q'].append(sum_q / frames)
        self._q_series['real_q'].append(rewards / frames)

        self._q_series_src.stream({attr: seq[-1] for attr, seq in self._q_series.items()})


__server = None  # type: Union[None, Server]
__server_thread = None  # type: Union[Thread, None]
__server_lock = Lock()


def serve_visualization():
    global __server
    global __server_thread
    with __server_lock:
        if __server is None:
            __server = Server({'/': VisApp})
            __server.start()
            print('Opening Bokeh application on http://localhost:5006/')
            __server.io_loop.add_callback(__server.show, "/")
            __server_thread = Thread(target=__server.io_loop.start)
            __server_thread.start()
        else:
            print('Cannot start bokeh server twice. It should already be running on http://localhost:5006/')


def unserve_visualization():
    global __server
    global __server_thread
    with __server_lock:
        if __server is not None:
            __server.io_loop.add_callback(__server.io_loop.stop)
            __server_thread.join()
            __server.stop()
            __server = None
            __server_thread = None
            print(print("Bokeh server stopped."))
        else:
            print("Cannot stop bokeh server because it was not started.")


_STANDALONE = False


if __name__ == '__main__':
    _STANDALONE = True
    from Learning import game_loop, make_env
    serve_visualization()
