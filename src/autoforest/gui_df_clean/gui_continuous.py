from autoforest.gui_df_clean.st_api import *
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np

__all__ = ['show_continuous_stats']
PLOT_FONT_SIZE = 4
PLOT_NUM_BINS = 60


# todo, show stats, min, max, skew, std, mean, median
def show_operations(df: pd.DataFrame, label):
    # normalize
    # log
    # exp
    # replace
    with st.expander('Operations'):
        if 'float' in df[label].dtype.name:
            options = ['log', 'exp', 'normalize', 'reset_ops']
        else:
            options = ['replace', 'reset_ops']

        s = st.selectbox(f"Apply operation", options=options)
        if s == 'log':
            df[label] = np.log(df[label])
        elif s == 'exp':
            df[label] = np.exp(df[label])
        elif s == 'normalize':
            std = df[label].std()
            mean = df[label].mean()
            df[label] = (df[label] - mean) / std
        elif s == 'replace':
            with st.form("replace value"):
                target = st.text_input('Replace value:')
                new_value = st.text_input('with:')
                submitted = st.form_submit_button("Replace")
                if submitted:
                    target = int(target)
                    new_value = int(new_value)
                    mask = df[label] == target
                    repl_mask = "{label}_replaces"
                    if repl_mask in df.columns:
                        mask = df[repl_mask] | mask
                    # add column in dataframe telling which values we have replaced
                    df[repl_mask] = mask
                    df.loc[mask, label] = new_value


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
