

from autoforest.gui_df_clean.constants import DEFAULT_PLOT_SPACING, DEFAULT_PLOT_LINEWIDTH, DEFAULT_FONT_DICT
from autoforest.gui_df_clean.st_api import *
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st
from autoforest.gui_df_clean.gui_helpers import *

__all__ = ['show_continuous_stats']
PLOT_FONT_SIZE = 4
PLOT_NUM_BINS = 60


def show_continuous_stats():
    df = get_df()
    label = get_label()

    if 'float' in df[label].dtype.name:
        options = [' ', 'log', 'exp', 'normalize', 'add', 'drop']
    else:
        options = [' ', 'replace', 'add', 'drop']
    show_operations(df, label, options)
    font = {'size': PLOT_FONT_SIZE}
    matplotlib.rc('font', **font)

    fig = plt.figure(figsize=(6, 4))
    plt.subplot(2, 1, 1)

    plt.title('Histogram', fontdict=DEFAULT_FONT_DICT)
    plt.hist(df[label], bins=PLOT_NUM_BINS)

    plt.subplot(2, 1, 2)
    plt.title('Values', fontdict=DEFAULT_FONT_DICT)
    df[label].plot(linewidth=DEFAULT_PLOT_LINEWIDTH)
    fig.tight_layout(pad=DEFAULT_PLOT_SPACING)

    st.write(plt.gcf())
