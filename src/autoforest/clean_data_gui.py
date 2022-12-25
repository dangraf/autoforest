import streamlit as st
from enum import Enum
import pandas as pd
from pathlib import Path

from autoforest.clean_data import df_shrink
__all__ = ['CleanState',
           'DataCleanerGui']

class CleanState(Enum):
    SEL_FILE = 0
    ITERATE_COLUMNS = 1


class DataCleanerGui:
    def start_gui(self):
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:

            p = Path(uploaded_file.name)
            print(p.suffix)
            if p.suffix == '.csv':
                df = pd.read_csv(uploaded_file, index_col=[0])
            elif p.suffix == '.json':
                df = pd.read_json(uploaded_file)
            elif p.suffix == '.pcl':
                df = pd.read_picke(uploaded_file)
            # st.session_state['df'] = df_shrink(df)
            st.session_state['df'] = df
            st.session_state['state'] = CleanState.ITERATE_COLUMNS

            # todo, check file ending and select correct function to read the file

    def iterate_columns(self):
        df = st.session_state['df']
        col_index = st.session_state['col_index']
        col = df.columns[col_index]
        st.write(f"Column Name: {df.columns[st.session_state['col_index']]}")
        t = df[col].dtype
        st.write(f"ColIndex type: {str(t)}")
        st.selectbox(label='Column Type:', options=['float', 'int', 'categorical', 'datetime'])
        if st.button('next'):
            st.session_state['col_index'] += 1
            if st.session_state['col_index'] > len(df):
                st.session_state['col_index'] = len(df)

        if st.button('prev'):
            st.session_state['col_index'] -= 1
            if st.session_state['col_index'] < 0:
                st.session_state['col_index'] = 0
        if st.button('exit and save'):
            print('exiting')

    def run_state_machine(self):
        if 'state' not in st.session_state:
            st.session_state['state'] = CleanState.SEL_FILE
        if 'col_index' not in st.session_state:
            st.session_state['col_index'] = 0

        if st.session_state['state'].value == CleanState.SEL_FILE.value:
            self.start_gui()
        if st.session_state['state'].value == CleanState.ITERATE_COLUMNS.value:
            self.iterate_columns()


if __name__ == "__main__":
    cleaner = DataCleanerGui()
    cleaner.run_state_machine()
