import streamlit as st
import pandas as pd
from enum import Enum

__all__ = ['get_label',
           'get_df',
           'set_df',
           'get_col_index',
           'set_col_index',
           'set_state',
           'get_state',
           'CleanState',
           'init_states',
           'set_backup_df',
           'get_backup_df',
           'get_col_type',
           'add_operation',
           'get_operations',
           'replace_operation',
           'clear_operations']

from autoforest.clean_data import NormalizedDtype


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


def get_label() -> str:
    return st.session_state['df'].columns[st.session_state['col_index']]


def get_df() -> pd.DataFrame:
    return st.session_state['df']


def set_df(df: pd.DataFrame):
    st.session_state['df'] = df


def get_state() -> CleanState:
    return st.session_state['state']


def set_state(state: CleanState):
    st.session_state['state'] = state


def set_backup_df(df: pd.DataFrame):
    st.session_state['df_backup'] = df


def get_backup_df() -> pd.DataFrame:
    return st.session_state['df_backup']


def get_col_type() -> NormalizedDtype:
    label = get_label()
    df = get_df()
    return NormalizedDtype.get_normalized_dtype(df[label].dtype)


def add_operation(obj, label):
    print(f'adding:{label}, {obj.name}')
    ops = get_operations()
    print(f"len ops: {len(ops)}")
    ops.append(obj)
    st.session_state['operations'][label] = ops


def get_operations():
    label = get_label()
    return st.session_state['operations'].get(label, [])


def replace_operation(obj):
    label = get_label()
    ops = get_operations()
    if len(ops) > 0 and ops[-1].name == obj.name:
        ops[-1] = obj
    else:
        add_operation(obj, label)


def clear_operations():
    label = get_label()
    st.session_state['operations'][label] = list()


def init_states():
    if 'state' not in st.session_state:
        st.session_state['state'] = CleanState.SEL_FILE
    if 'col_index' not in st.session_state:
        st.session_state['col_index'] = -1
    if 'df' not in st.session_state:
        st.session_state['df'] = pd.DataFrame()
    if 'df_backup' not in st.session_state:
        print('adding df_backup')
        st.session_state['df_backup'] = pd.DataFrame()
    if 'operations' not in st.session_state:
        print('adding operations')
        st.session_state['operations'] = dict()
