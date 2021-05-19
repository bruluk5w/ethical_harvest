import os
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from itertools import groupby, islice
from threading import Thread, Lock
from typing import Union, List

from bokeh.document import Document
from bokeh.layouts import column
from bokeh.models import Dropdown
from bokeh.server.server import Server

from constants import get_model_name_params
from impl.config import get_experiment_names, get_storage, cfg, load_cfg, set_config
from impl.stats_to_file import StatsWriter
from impl.vis.stats_panel import StatsPanel


class VisApp:

    def __init__(self, doc: Document):
        if _STANDALONE:
            experiment_names = get_experiment_names()
            self._name = experiment_names[0] if experiment_names else None  # name of the selected experiment
            self._experiment_dropdown = Dropdown(label="No experiments available" if self._name is None else self._name,
                                                 menu=experiment_names)
            self._experiment_dropdown.on_click(self._on_experiment_change)
        else:
            self._name = cfg().EXPERIMENT_NAME
            assert self._name is not None
            self._experiment_dropdown = Dropdown(label=self._name)

        self._storage = None  # folder for all the data of the experiment
        self._thread_pool = ThreadPoolExecutor()
        self._stats_panel = StatsPanel(doc, self._thread_pool)

        doc.add_root(column(self._experiment_dropdown, self._stats_panel.get_layout()))

        self._load_experiment(self._name)

    def _on_experiment_change(self, event):
        self._experiment_dropdown.label = event.item
        self._load_experiment(event.item)

    def _load_experiment(self, name):
        global _STANDALONE

        if name is None:
            return
        self._name = name
        self._stats_panel.set_experiment(self._name)
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

        checkpoint_params = (get_model_name_params(os.path.splitext(name)[0]) for name in
                             next(os.walk(os.path.join(self._storage, 'weights')))[2])
        episode_idx_key = lambda params: params[0]
        checkpoint_params = sorted((params for params in checkpoint_params if params[0] is not None),
                                   key=episode_idx_key)
        frames = ((key, *islice(zip(*group), 1, None)) for key, group in groupby(checkpoint_params, key=episode_idx_key))

        # Start the load operations and mark each future with its URL
        self.future_to_frame = {
            self.threadPool.submit(self._load_episode_frame, episode_idx, agent_indices, model_types): episode_idx
            for episode_idx, agent_indices, model_types in frames
        }

        # for episode_idx, agent_indices, model_types in frames:
        #
        #     self._load_episode_frame(episode_idx, agent_indices, model_types)

    def _load_episode_frame(self, episode_idx: int, agent_indices: List[int], model_types: List[str]):
        agent_indices = sorted(k for k, _ in groupby(agent_indices))
        if agent_indices != list(range(len(agent_indices))) or len(agent_indices) != cfg().NUM_AGENTS:
            print("Saved models for episode {} missing. Skipping frame.".format(episode_idx))
            return
        acc = StatsWriter()
        game_loop(make_env(), episodes=episode_idx + 1, train=False, episode_callback=acc, start_episode=episode_idx, verbose=False)

        self._q_series['episode'].append(episode_idx)
        self._q_series['q_max'].append(acc.max_q)
        self._q_series['q_min'].append(acc.min_q)
        self._q_series['q_avg'].append(acc.sum_q / acc.frames)
        self._q_series['real_q'].append(acc.rewards / acc.frames)

        self._q_series_src.stream({attr: [seq[-1]] for attr, seq in self._q_series.items()})


__server = None  # type: Union[None, Server]
__server_thread = None  # type: Union[Thread, None]
__server_lock = Lock()


def serve_visualization(port=5006):
    global __server
    global __server_thread
    with __server_lock:
        if __server is None:
            __server = Server({'/': VisApp}, port=port)
            __server.start()
            print('Opening Bokeh application on http://localhost:{}/'.format(port))
            __server.io_loop.add_callback(__server.show, "/")
            __server_thread = Thread(target=__server.io_loop.start, name="Server Thread")
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
    serve_visualization(port=5008)
