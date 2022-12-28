import streamlit as st
from enum import Enum

__all__ = ['get_label',
           'get_df',
           'set_df',
           'get_col_index',
           'set_col_index',
           'CleanState']


class CleanState(Enum):
    SEL_FILE = 0
    ITERATE_COLUMNS = 1


def get_col_index():
    return st.session_state['col_index']


def set_col_index(index: int):
    df = get_df()
    if index < 0:
        index = 0
    if index >= len(df.columns):
        index = len(df.columns) - 1
    st.session_state['col_index'] = index


def get_label():
    return st.session_state['df'].columns[st.session_state['col_index']]


def get_df():
    return st.session_state['df']


def set_df(df):
    st.session_state['df'] = df
