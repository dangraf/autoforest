import streamlit as st
from enum import Enum
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from autoforest.clean_data import *

__all__ = ['CleanState',
           'DataCleanerGui']


class CleanState(Enum):
    SEL_FILE = 0
    ITERATE_COLUMNS = 1

def read_dataframe(uploaded_file):
    p = Path(uploaded_file.name)
    print(p.suffix)
    if p.suffix == '.csv':
        df = pd.read_csv(uploaded_file, index_col=[0])
    elif p.suffix == '.json':
        df = pd.read_json(uploaded_file)
    elif p.suffix == '.pcl':
        df = pd.read_picke(uploaded_file)
    return df

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


class DataCleanerGui:
    def start_gui(self):
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            df = read_dataframe(uploaded_file)
            st.session_state['df'] = df_shrink(df)
            st.session_state['state'] = CleanState.ITERATE_COLUMNS
            st.session_state['col_index'] = 0
            st.experimental_rerun()

    def iterate_columns(self):
        df = st.session_state['df']
        col_index = st.session_state['col_index']

        print(col_index)
        col = df.columns[col_index]
        st.write(f"Column Name: {df.columns[st.session_state['col_index']]}")
        t = df[col].dtype
        st.write(f"ColIndex type: {str(t)}")
        # st.selectbox(label='Column Type:', options=['float', 'int', 'categorical', 'datetime'])
        if is_cat(df, col):
            show_categorigal_stats(df, col)
        print(f'saving df {col}')
        st.session_state['df'] = df
        if st.button('next'):
            print('next')
            st.session_state['col_index'] += 1
            if st.session_state['col_index'] > len(df):
                st.session_state['col_index'] = len(df)
            st.experimental_rerun()

        if st.button('prev'):
            print('prev')
            st.session_state['col_index'] -= 1
            if st.session_state['col_index'] < 0:
                st.session_state['col_index'] = 0
            st.experimental_rerun()






    def run_state_machine(self):
        if 'state' not in st.session_state:
            st.session_state['state'] = CleanState.SEL_FILE
        if 'col_index' not in st.session_state:
            st.session_state['col_index'] = -1

        if st.session_state['state'].value == CleanState.SEL_FILE.value:
            self.start_gui()
        if st.session_state['state'].value == CleanState.ITERATE_COLUMNS.value:
            self.iterate_columns()


if __name__ == "__main__":
    cleaner = DataCleanerGui()
    cleaner.run_state_machine()
