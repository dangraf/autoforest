from statsmodels.tsa.stattools import adfuller, kpss
from autoforest.gui_df_clean.st_api import *
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
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
               'add': TfmAdd,
               'diff': TfmDiff,
               'drop': DropCol}
    #with st.expander('Operations'):
    if 'float' in df[label].dtype.name:
        options = [' ', 'log', 'exp', 'normalize', 'add', 'drop']
    else:
        options = [' ', 'replace', 'add', 'drop']

    s = st.selectbox(f"Apply operation", options=options)

    operation = op_list[s]
    if operation is not None:
        tfm = operation.show_form(stobj=st, df=None, label=label)
        if tfm:
            try:
                df = tfm.encodes(df)
                add_operation(tfm, label)
                set_df(df)
            except BaseException as e:
                print(f'Error{e}')


def get_adfuller_result(timeseries):
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    return dfoutput


def get_kpss_result(timeseries):
    kpsstest = kpss(timeseries, regression='c')
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])
    for key, value in kpsstest[3].items():
        kpss_output['Critical Value (%s)' % key] = value
    return kpss_output


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
    df[label].plot(linewidth=0.5)

    st.write(plt.gcf())
