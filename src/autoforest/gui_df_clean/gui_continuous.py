from autoforest.gui_df_clean.st_api import *
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
from autoforest.operations import *

__all__ = ['show_continuous_stats']
PLOT_FONT_SIZE = 4
PLOT_NUM_BINS = 60


# todo, show stats, min, max, skew, std, mean, median
def show_operations(df: pd.DataFrame, label):
    # normalize
    # log
    # exp
    # replace
    op_list = {' ': None,
               'log': TfmLog,
               'exp': TfmExp,
               'normalize': TfmNormalize,
               'replace': TfmReplace,
               'add': TfmAdd}
    with st.expander('Operations'):
        if 'float' in df[label].dtype.name:
            options = [' ', 'log', 'exp', 'normalize', 'add']
        else:
            options = [' ', 'replace', 'add']

        s = st.selectbox(f"Apply operation", options=options)

        operation = op_list[s]
        if operation is not None:
            tfm = operation.show_form(stobj=st, label=label)
            if tfm:
                try:
                    df = tfm.encodes(df)
                    add_operation(tfm)
                    set_df(df)
                except BaseException as e:
                    print(f'Error{e}')

def show_continuous_stats():
    df = get_df()
    label = get_label()
    font = {'size': PLOT_FONT_SIZE}
    matplotlib.rc('font', **font)

    show_operations(df, label)

    plt.figure(figsize=(6, 2))
    plt.subplot(2, 1, 1)
    plt.title('histogram')
    plt.hist(df[label], bins=PLOT_NUM_BINS)

    plt.subplot(2, 1, 2)
    plt.title('values')
    df[label].plot()

    st.pyplot(plt.gcf())
