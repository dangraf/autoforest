import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from autoforest.clean_data import get_na_mask
from autoforest.gui_df_clean.st_api import *

__all__ = ['reorder_categories',
           'show_categorigal_stats']

PLOT_MAX_NUM_CATEGORIES = 120
PLOT_FONT_SIZE = 4
MAX_LEN_REORDER = 25


def reorder_categories(df: pd.DataFrame, label):
    num_cats = len(df[label].cat.categories)
    options = list(range(num_cats))
    selections = list()
    categories = list(df[label].cat.categories)
    if len(categories) > MAX_LEN_REORDER:
        # datetimes etc can be categorical but seldom need to be ordered
        return
    cats = ', '.join(list(df[label].cat.categories))
    st.write(f"**Current order of categories:** \n {cats}")
    with st.expander('Re-order Categories'):
        for i, cat in enumerate(categories):
            s = st.selectbox(f"{cat}", options=options, index=i)
            selections.append(s)
        for i, sel in enumerate(selections):
            if i != sel:
                categories[sel], categories[i] = categories[i], categories[sel]
                df[label] = df[label].cat.reorder_categories(categories, ordered=True)
                st.experimental_rerun()
                break


def show_categorigal_stats():
    df = get_df()
    label = get_label()
    font = {'size': PLOT_FONT_SIZE}

    matplotlib.rc('font', **font)

    if df[label].dtype.name == 'category':
        reorder_categories(df, label)
    plt.subplot(4, 1, 1)
    plt.title('Count Values')
    df[label].value_counts()[:PLOT_MAX_NUM_CATEGORIES].plot(kind='bar', figsize=(6, 3))
    plt.xticks(rotation=45, ha='right')
    plt.subplot(4, 1, 3)
    plt.title('Categories converted to values')
    df[label].cat.codes.plot()

    nan_mask = get_na_mask(df, label)
    if nan_mask.sum() != 0:
        plt.subplot(4, 1, 4)
        plt.title('NaN values')
        plt.plot(nan_mask)
    fig = plt.gcf()
    st.write(fig)
    st.write(f"Num nan: {df[label].isna().sum()}")
    return df
