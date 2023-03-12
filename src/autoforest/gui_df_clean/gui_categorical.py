import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from autoforest.clean_data import get_na_mask
from autoforest.gui_df_clean.constants import DEFAULT_PLOT_SPACING, DEFAULT_PLOT_LINEWIDTH, DEFAULT_FONT_DICT
from autoforest.gui_df_clean.st_api import *
from autoforest.gui_df_clean.gui_helpers import show_operations

__all__ = ['show_categorigal_stats']

PLOT_MAX_NUM_CATEGORIES = 120
PLOT_FONT_SIZE = 4
MAX_LEN_REORDER = 25


def show_categorigal_stats():
    df = get_df()
    label = get_label()
    font = {'size': PLOT_FONT_SIZE}

    matplotlib.rc('font', **font)


    nan_mask = get_na_mask(df, label)

    show_operations(df=df, label=label, options=['reorder cats', 'drop'])
    plt.subplot(3, 1, 1)
    plt.title('Count Values', fontdict=DEFAULT_FONT_DICT)
    if nan_mask.sum() != 0:
        figsize_y = 5
    else:
        figsize_y = 4
    df[label].value_counts()[:PLOT_MAX_NUM_CATEGORIES].plot(kind='bar', figsize=(6, figsize_y), linewidth=1.5)
    plt.xticks(rotation=45, ha='right')
    plt.subplot(3, 1, 2)

    plt.title('Categories converted to values', fontdict=DEFAULT_FONT_DICT)
    df[label].cat.codes.plot(linewidth=DEFAULT_PLOT_LINEWIDTH)

    if nan_mask.sum() != 0:
        plt.subplot(3, 1, 3)
        plt.title('NaN values', fontdict=DEFAULT_FONT_DICT)
        plt.plot(nan_mask, linewidth=DEFAULT_PLOT_LINEWIDTH)
    fig = plt.gcf()
    fig.tight_layout(pad=DEFAULT_PLOT_SPACING)
    st.write(fig)
    st.write(f"Num nan: {df[label].isna().sum()}")
    return df
