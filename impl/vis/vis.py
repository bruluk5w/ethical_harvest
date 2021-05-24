from threading import Thread, Lock
from typing import Union

from bokeh.document import Document
from bokeh.layouts import column
from bokeh.models import Dropdown
from bokeh.server.server import Server

from impl.config import get_experiment_names, cfg
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
        self._stats_panel = StatsPanel(doc)

        doc.add_root(column(self._experiment_dropdown, self._stats_panel.get_layout(),
                            margin=(0, 20, 0, 0), sizing_mode='scale_width'))

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
    serve_visualization(port=5009)
