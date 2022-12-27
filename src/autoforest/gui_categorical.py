import streamlit as st
import pandas as pd
from autoforest.clean_data import *
import matplotlib.pyplot as plt
import matplotlib

__all__ = ['reorder_categories',
           'show_categorigal_stats']


def reorder_categories(df: pd.DataFrame, label):
    num_cats = len(df[label].cat.categories)
    options = list(range(num_cats))
    selections = list()
    categories = list(df[label].cat.categories)
    if len(categories) > 40:
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


def show_categorigal_stats(df: pd.DataFrame, label):
    font = {'size': 4}

    matplotlib.rc('font', **font)

    if df[label].dtype.name == 'category':
        reorder_categories(df, label)


    df[label].value_counts()[:120].plot(kind='bar', figsize=(6, 1))
    fig = plt.gcf()
    st.pyplot(fig)
    st.write(f"Num nan: {df[label].isna().sum()}")
    return df
