from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import partial
from typing import Callable

from bokeh.document import without_document_lock
from tornado import gen

from impl.stats import build_summary, Stats


class DataSink:
    def __init__(
            self, doc,
            thread_pool: ThreadPoolExecutor,
            update_callback: Callable[[Stats], None]
    ):
        self._target_doc = doc
        self._update_callback = update_callback
        self._thread_pool = thread_pool
        self._last_frame_idx = 0
        self._last_episode_idx = 0

    def on_new_data(self, series: Stats):
        s = deepcopy(series)
        self._target_doc.add_next_tick_callback(partial(self._build, s))

    @gen.coroutine
    @without_document_lock
    def _build(self, s: Stats):
        yield self._thread_pool.submit(build_summary, s)
        self._target_doc.add_next_tick_callback(partial(self._update_callback, s))
