from concurrent.futures import ThreadPoolExecutor
from typing import Union

import numpy as np
from bokeh.document import Document
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Band
from bokeh.plotting import figure

from impl.config import get_storage_for_experiment
from impl.stats import Properties, SummaryProperties, Stats
from impl.stats_to_file import QStatsReader, get_trace_file_path
from impl.vis.data_sink import DataSink
import math


class StatsPanel:
    def __init__(self, doc: Document, thread_pool: ThreadPoolExecutor):
        self._doc = doc
        self._thread_pool = thread_pool

        self._episode_summary_plot = figure(x_axis_label='frames', y_axis_label='value_estimates',
                                            title="Size of q values")
        self._episode_summary_avg_q_line = None

        self._episode_summary_src = ColumnDataSource()
        self._episode_detail_src = ColumnDataSource()
        self._agent_summary_src = ColumnDataSource()
        self._agent_detail_src = ColumnDataSource()

        self._has_plot = False

        self._data_sink = DataSink(self._doc, self._thread_pool, self.update_data)


        # self._value_estimates_plot.quad(top='q_max', bottom='q_min',
        #                                left=transform('episode', shift_left), right=transform('episode', shift_right),
        #                                 source=self._q_series_src,
        #                                 color='red', alpha=0.5)
        # self._value_estimates_plot.line(x='episode', y='q_avg', source=self._q_series_src, color='red')
        # self._value_estimates_plot(x='episode', y=transform('real_q', mean),
        #                                 source=self._q_series_src,
        #                                 color='blue')

        self._stats_reader = None  # type:Union[None, QStatsReader]

    def get_layout(self):
        # put the results in a column and show
        return column(self._episode_summary_plot)

    def set_experiment(self, experiment_name):
        if self._stats_reader is not None:
            self._stats_reader.stop()
            self._stats_reader = None

        if experiment_name is not None:
            path = get_storage_for_experiment(experiment_name)
            self._stats_reader = QStatsReader(
                get_trace_file_path(path),
                self._data_sink.on_new_data
            )

    def update_data(self, s: Stats):
        try:
            self._episode_summary_src.data = src = {
                SummaryProperties.MAX_Q.value: s.summary.max_q,
                SummaryProperties.MIN_Q.value: s.summary.min_q,
                SummaryProperties.AVG_Q.value: s.summary.avg_q,
                SummaryProperties.SUM_REWARD.value: s.summary.sum_reward,
                SummaryProperties.EPISODE_START.value: s.summary.episode_start,
            }
            # self._episode_detail_src = {
            #     Properties.LAST_Q
            # }

            if not self._has_plot:
                # create plots when the data arrives for the first time
                self._episode_summary_avg_q_line = self._episode_summary_plot.line(
                    x=SummaryProperties.EPISODE_START.value,
                    y=SummaryProperties.AVG_Q.value,
                    source=self._episode_summary_src
                )
                band = Band(base=SummaryProperties.EPISODE_START.value,
                            lower=SummaryProperties.MIN_Q.value,
                            upper=SummaryProperties.MAX_Q.value,
                            source=self._episode_summary_src,
                            level='underlay', fill_alpha=1.0, line_width=1, line_color='gray')
                self._episode_summary_plot.add_layout(band)
                self._has_plot = True

        except Exception as e:
            raise

