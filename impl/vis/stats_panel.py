from functools import partial
from itertools import cycle, chain, permutations
from textwrap import wrap
from typing import Union, Callable, List

import numpy as np
from bokeh import palettes
from bokeh.colors import named as color, RGB
from bokeh.core.enums import ButtonType
from bokeh.document import Document
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Band, CustomJS, Toggle, Slider
from bokeh.plotting import figure
from scipy.ndimage import convolve

from impl.config import get_storage_for_experiment
from impl.stats import SummaryProperties, Stats, Properties
from impl.stats_to_file import QStatsReader, get_trace_file_path
from impl.vis.data_sink import DataSink


def gini(series: List[np.ndarray]):
    if not series:
        return None

    total = np.zeros_like(series[0], dtype=np.float32)
    for i in range(len(series)):
        total = np.add(total, series[i], out=total)

    np.multiply(total, 2 * len(series), out=total)

    result = np.zeros_like(series[0], dtype=np.float32)
    intermediate = np.zeros_like(series[0])
    for i, j in permutations(range(len(series)), 2):
        np.subtract(series[i], series[j], out=intermediate)
        np.abs(intermediate, out=intermediate)
        np.add(result, intermediate, out=result)

    np.divide(result, total, out=result)
    return result


def mean_log_deviation(series):
    if not series:
        return None

    arr = np.vstack(series).astype(np.float32)
    np.add(arr, 0.005, out=arr)  # to avoid -inf for x=0
    log_of_mean = np.empty_like(series[0], dtype=np.float32)
    np.mean(arr, axis=0, out=log_of_mean)
    np.log(log_of_mean, dtype=np.float32, out=log_of_mean)

    mean_of_log = np.empty_like(series[0], dtype=np.float32)
    log = np.log(arr, dtype=np.float32)
    np.mean(log, axis=0, out=mean_of_log)

    np.subtract(log_of_mean, mean_of_log, out=mean_of_log)
    return mean_of_log


class StatsPanel:
    def __init__(self, doc: Document):
        self._doc = doc
        self._stats = None
        self._smooth_size = Slider(start=2, end=50, value=3, step=1, title='Window Size')
        self._toggle_smooth_btn = Toggle(label='Smoothing Off', active=False)
        self._toggle_smooth_btn.on_click(self._toggle_smooth)
        self._smooth_size.on_change('value_throttled', self._set_smoothing_window_size)
        self._toggle_smooth()

        self._all_plots = (
            figure(x_axis_label='frames', y_axis_label='p', title='explore_probability', height=200, sizing_mode='scale_width'),
            figure(x_axis_label='frames', y_axis_label='q estimate', title='Q estimates per episode (\"all\" is mean)', height=300, sizing_mode='scale_width'),
            figure(x_axis_label='frames', y_axis_label='reward', title='Summed rewards per episode (\"all\" is mean)', height=300, sizing_mode='scale_width'),
            figure(x_axis_label='frames', y_axis_label='num_apples', title='Apples at end of episode (\"all\" is sum)', height=300, sizing_mode='scale_width'),
            figure(x_axis_label='frames', y_axis_label='num_apples', title='Average apples per episode', height=300, sizing_mode='scale_width'),
            figure(x_axis_label='frames', y_axis_label='num_apples', title='Donations per episode', height=300, sizing_mode='scale_width'),
            figure(x_axis_label='frames', y_axis_label='G', title='Equality at end of episode by number of owned apples', height=300, sizing_mode='scale_width'),
        )

        self._p_explore_plot, self._q_plot, self._reward_plot, self._apple_end_plot, self._apple_avg_plot, \
            self._donations_plot, self._gini_plot = self._all_plots

        # link x range
        last_plot = self._all_plots[0]
        for plot in self._all_plots[1:]:
            plot.x_range = last_plot.x_range

        self._episode_summary_src = ColumnDataSource()

        self._has_plot = False

        self._data_sink = DataSink(self._doc, self.update_data)
        self._stats_reader = None  # type:Union[None, QStatsReader]

    def get_layout(self):
        return column(list(chain(
            [self._toggle_smooth_btn, self._smooth_size],
            list(self._all_plots)
        )), sizing_mode='scale_width')

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

    def _toggle_smooth(self, is_smooth=None):
        if is_smooth is None:
            is_smooth = self._toggle_smooth_btn.active

        self._toggle_smooth_btn.label = 'Smoothing On' if is_smooth else 'Smoothing Off'
        self._toggle_smooth_btn.button_type = ButtonType.default if is_smooth else ButtonType.light
        if self._stats is not None:
            self.update_data(self._stats)

    def _set_smoothing_window_size(self, attr, old, new):
        if self._stats is not None and self._toggle_smooth_btn.active:
            self.update_data(self._stats)

    def update_data(self, s: Stats):
        self._stats = s
        try:
            t = self._transform
            src = {
                SummaryProperties.MAX_Q.value: t(s.summary.max_q),
                SummaryProperties.MIN_Q.value: t(s.summary.min_q),
                SummaryProperties.AVG_Q.value: t(s.summary.avg_q),
                SummaryProperties.SUM_REWARD.value: t(s.summary.sum_reward),
                SummaryProperties.MAX_OWNED_APPLES.value: t(s.summary.max_owned_apples),
                SummaryProperties.END_OWNED_APPLES.value: t(s.summary.end_owned_apples),
                SummaryProperties.AVG_OWNED_APPLES.value: t(s.summary.avg_owned_apples),
                SummaryProperties.TOTAL_DONATED_APPLES.value: t(s.summary.total_donated_apples),
                SummaryProperties.EPISODE_START.value: s.summary.episode_start,
                Properties.LAST_EXPLORE_PROBABILITY.value: s.agent_series[0].last_explore_probability[s.summary.episode_start],
                'all_apples': t(s.summary.end_owned_apples + s.summary.total_donated_apples),
                'mld': t(mean_log_deviation([s.agent_series[i].summary.end_owned_apples for i in range(len(s.agent_series))])),
                'gini': t(gini([s.agent_series[i].summary.end_owned_apples for i in range(len(s.agent_series))])),
            }

            agent_key = lambda idx, property: 'agent_{}_{}'.format(idx, property.value)

            for idx, series in enumerate(s.agent_series):
                src[agent_key(idx, SummaryProperties.MAX_Q)] = t(series.summary.max_q)
                src[agent_key(idx, SummaryProperties.MIN_Q)] = t(series.summary.min_q)
                src[agent_key(idx, SummaryProperties.AVG_Q)] = t(series.summary.avg_q)
                src[agent_key(idx, SummaryProperties.SUM_REWARD)] = t(series.summary.sum_reward)
                src[agent_key(idx, SummaryProperties.MAX_OWNED_APPLES)] = t(series.summary.max_owned_apples)
                src[agent_key(idx, SummaryProperties.END_OWNED_APPLES)] = t(series.summary.end_owned_apples)
                src[agent_key(idx, SummaryProperties.AVG_OWNED_APPLES)] = t(series.summary.avg_owned_apples)
                src[agent_key(idx, SummaryProperties.TOTAL_DONATED_APPLES)] = t(series.summary.total_donated_apples)
                src[agent_key(idx, SummaryProperties.EPISODE_START)] = series.summary.episode_start

            self._episode_summary_src.data = src

            if not self._has_plot:
                # create plots when the data arrives for the first time
                self._p_explore_plot.line(
                    x=SummaryProperties.EPISODE_START.value,
                    y=Properties.LAST_EXPLORE_PROBABILITY.value,
                    color=color.black,
                    source=self._episode_summary_src
                )

                self._apple_end_plot.line(
                    x=SummaryProperties.EPISODE_START.value,
                    y='all_apples',
                    color=color.green,
                    source=self._episode_summary_src,
                    line_width=2,
                    muted_alpha=0.1,
                    legend_label='all incl. pool',
                )

                self._gini_plot.line(
                    x=SummaryProperties.EPISODE_START.value,
                    y='gini',
                    color=color.blue,
                    source=self._episode_summary_src,
                    line_width=2,
                    muted_alpha=0.1,
                    legend_label='Gini Index',
                )

                self._gini_plot.line(
                    x=SummaryProperties.EPISODE_START.value,
                    y='mld',
                    color=color.black,
                    source=self._episode_summary_src,
                    line_width=2,
                    muted_alpha=0.1,
                    legend_label='Mean Log Deviation',
                )

                self._make_summary_plot('all', color.red, color.salmon, color.darksalmon, lambda p: p.value)

                for idx, base_color in zip(range(len(s.agent_series)), cycle(reversed(palettes.Colorblind8))):
                    base_color = RGB(*(int(component, 16) for component in wrap(base_color[1:], 2)))
                    self._make_summary_plot(
                        'agent {}'.format(idx),
                        line_color=base_color,
                        band_fill_color=base_color.lighten((1.0 - base_color.to_hsl().l) * 0.8),
                        band_line_color=base_color.lighten((1.0 - base_color.to_hsl().l) * 0.5),
                        src_key=partial(agent_key, idx),
                    )

                for plot in self._all_plots:
                    if len(plot.legend) > 0:
                        plot.legend.location = 'top_left'
                        plot.legend.click_policy = 'mute'

                self._has_plot = True

        except Exception as e:
            raise

    def _transform(self, arr: np.ndarray):
        if self._toggle_smooth_btn.active:
            window_size = self._smooth_size.value
            return convolve(arr.astype(np.float32, copy=False), np.ones(window_size)/window_size)

        return arr

    def _make_summary_plot(self, name, line_color, band_fill_color, band_line_color, src_key: Callable[[SummaryProperties], str]):
        q_renderer = self._q_plot.line(
            x=src_key(SummaryProperties.EPISODE_START),
            y=src_key(SummaryProperties.AVG_Q),
            color=line_color,
            muted_alpha=0.1,
            source=self._episode_summary_src,
            legend_label='{}'.format(name),
        )

        q_band = Band(
            base=src_key(SummaryProperties.EPISODE_START),
            lower=src_key(SummaryProperties.MIN_Q),
            upper=src_key(SummaryProperties.MAX_Q),
            source=self._episode_summary_src,
            level='underlay',
            line_color=band_line_color,
            line_alpha=1.0,
            fill_color=band_fill_color,
            fill_alpha=0.7,
            line_width=1,
        )

        self._q_plot.add_layout(q_band)
        q_renderer.js_on_change('muted', CustomJS(args=dict(band=q_band), code='band.visible = !cb_obj.muted;'))

        self._reward_plot.line(
            x=src_key(SummaryProperties.EPISODE_START),
            y=src_key(SummaryProperties.SUM_REWARD),
            source=self._episode_summary_src,
            line_width=1,
            color=line_color,
            muted_alpha=0.1,
            legend_label='{}'.format(name),
        )

        self._apple_end_plot.line(
            x=src_key(SummaryProperties.EPISODE_START),
            y=src_key(SummaryProperties.END_OWNED_APPLES),
            color=line_color,
            muted_alpha=0.1,
            source=self._episode_summary_src,
            legend_label='owned by {} at end'.format(name),
        )

        avg_apples_renderer = self._apple_avg_plot.line(
            x=src_key(SummaryProperties.EPISODE_START),
            y=src_key(SummaryProperties.AVG_OWNED_APPLES),
            color=line_color,
            muted_alpha=0.1,
            source=self._episode_summary_src,
            legend_label='mean and max owned by {}'.format(name),
        )

        apples_band = Band(
            base=src_key(SummaryProperties.EPISODE_START),
            lower=0.0,
            upper=src_key(SummaryProperties.MAX_OWNED_APPLES),
            source=self._episode_summary_src,
            level='underlay',
            line_color=band_line_color,
            line_alpha=1.0,
            fill_color=band_fill_color,
            fill_alpha=0.7,
            line_width=1,
        )

        self._apple_avg_plot.add_layout(apples_band)
        avg_apples_renderer.js_on_change('muted', CustomJS(args=dict(band=apples_band), code='band.visible = !cb_obj.muted;'))

        self._donations_plot.line(
            x=src_key(SummaryProperties.EPISODE_START),
            y=src_key(SummaryProperties.TOTAL_DONATED_APPLES),
            color=line_color,
            muted_alpha=0.1,
            source=self._episode_summary_src,
            legend_label='{}'.format(name),
        )
