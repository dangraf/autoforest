import streamlit as st
from enum import Enum
import pandas as pd
from pathlib import Path
from autoforest.gui_categorical import *

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
        st.write(f"# {df.columns[st.session_state['col_index']]}")
        na_mask = get_na_mask(df, label)
        num_na = na_mask.sum()
        na_pct = num_na.sum() / len(df) * 100
        st.write(f" **dtype:** {df[label].dtype.name}, {na_pct:.2f}% NaN values")

        col1, col2 = st.columns(2)
        col1.write(df[~na_mask][label][:5])

        option = col2.selectbox('NA sampeling func',
                                options=['', 'Random Sampling', 'Median', 'drop rows NA'])
        if option == 'drop rows NA':
            df = df[~na_mask].reset_index(drop=True)
        elif option == 'Median':
            print('Median')
            fill_median(df, label)
        elif option == 'Random Sampling':
            print('Random Sampling')
            fill_random_sampling(df, label)

        type = st.selectbox(label='Column Type:',
                            options=NormalizedDtype.get_list_of_types(),
                            index=NormalizedDtype.get_index_from_dtype(df[label].dtype),
                            on_change=try_convert,
                            key='try_convert',
                            args=(label,))
        print(type)
        if is_cat(df, label):
            df = show_categorigal_stats(df, label)
        print(f'saving df {label} len {len(df)}')
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
    st.set_page_config(layout="wide")
    cleaner = DataCleanerGui()
    cleaner.run_state_machine()
