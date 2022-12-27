import streamlit as st
from enum import Enum
import pandas as pd
from pathlib import Path
from autoforest.gui_categorical import *
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


def try_convert(label):
    df = st.session_state['df']
    df[label] = NormalizedDtype.try_convert(column=df[label],
                                            stype=st.session_state.try_convert)


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
        label = df.columns[col_index]
        st.sidebar.write(f"# {df.columns[st.session_state['col_index']]}")
        na_mask = get_na_mask(df, label)
        num_na = na_mask.sum()
        na_pct = num_na.sum() / len(df) * 100
        st.sidebar.write(f" **dtype:** {df[label].dtype.name}, {na_pct:.2f}% NaN values")

        try:
            st.sidebar.dataframe(df[~na_mask][label].iloc[:5])
        except BaseException as e:
            st.write(f"Error plotting dataframe: {e}")
        if num_na != 0:
            st.write(
                '## Missing values\n',
                'Please use sampling methods below to fill missing values, note the "drop-na" method cant be undone')
            option = st.selectbox('NA sampeling func',
                                  options=['', 'Random Sampling', 'Median', 'drop rows NA'])
            if option == 'drop rows NA':
                df = df[~na_mask].reset_index(drop=True)
            elif option == 'Median':
                fill_median(df, label)
            elif option == 'Random Sampling':
                fill_random_sampling(df, label)

        st.sidebar.selectbox(label='Column Type:',
                             options=NormalizedDtype.get_list_of_types(),
                             index=NormalizedDtype.get_index_from_dtype(df[label].dtype),
                             on_change=try_convert,
                             key='try_convert',
                             args=(label,))
        ntype = NormalizedDtype.get_normalized_dtype(df[label].dtype)
        print(ntype.value)
        if ntype.value == NormalizedDtype.Categorical.value:
            print('Enter Categorical')
            df = show_categorigal_stats(df, label)
        else:
            try:
                df[label].plot()
                st.pyplot(plt.gcf())
            except BaseException as e:
                st.write(f"error plotting: {e}")

        st.session_state['df'] = df
        col1, col2 = st.sidebar.columns(2)
        if col2.button('next'):
            print('next')
            st.session_state['col_index'] += 1
            if st.session_state['col_index'] > len(df):
                st.session_state['col_index'] = len(df)
            st.experimental_rerun()

        if col1.button('prev'):
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
    st.set_page_config(layout="wide")
    cleaner = DataCleanerGui()
    cleaner.run_state_machine()
