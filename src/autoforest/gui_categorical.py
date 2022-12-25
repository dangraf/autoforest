import streamlit as st
import pandas as pd
from autoforest.clean_data import *
import matplotlib.pyplot as plt


def reorder_categories(df: pd.DataFrame, label):
    num_cats = len(df[label].cat.categories)
    options = list(range(num_cats))
    selections = list()
    categories = list(df[label].cat.categories)
    for i, cat in enumerate(categories):
        s = st.selectbox(f"{cat}", options=options, index=i)
        selections.append(s)
    for i, sel in enumerate(selections):
        if i != sel:
            categories[sel], categories[i] = categories[i], categories[sel]
            df[label] = df[label].cat.reorder_categories(categories, ordered=True)
            st.experimental_rerun()
            break


def show_categorigal_stats(df, label):
    num_na = df[label].isna().sum()
    na_pct = num_na / len(df)
    st.write(f"{na_pct:.2f}% of data contains NaN values")
    if df[label].dtype.name == 'category':
        reorder_categories(df, label)

    if num_na != 0:
        st.write("## NaN values")
        st.write(
            "NaN values need to be handled, an extra na-mask column is created to tell which values that are generated")
        option = st.selectbox('Sampling func',
                              options=['Random Sampling', 'Median', 'drop NA'])
        if option == 'drop rows NA':
            print('drop NA')
            df.dropna(subset=[label], inplace=True)
        elif option == 'Median':
            print('Median')
            fill_median(df, label)
        elif option == 'Random Sampling':
            print('Random Sampling')
            fill_random_sampling(df, label)

    df[label].value_counts().plot(kind='bar')

    fig = plt.gcf()
    st.pyplot(fig)
    st.write(f"Num nan: {df[label].isna().sum()}")
    st.session_state['df'] = df
