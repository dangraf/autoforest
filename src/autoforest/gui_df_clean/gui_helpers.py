import pandas as pd
import streamlit as st
from typing import List
from autoforest.operations import *
from autoforest.gui_df_clean.st_api import *

__all__ = ['show_operations']


def show_operations(df: pd.DataFrame, label: str, options: List[str]):
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
               'drop': DropCol,
               'reorder cats': ReorderCategories}
    # with st.expander('Operations'):
    # if 'float' in df[label].dtype.name:
    #    options = [' ', 'log', 'exp', 'normalize', 'add', 'drop']
    # else:
    #    options = [' ', 'replace', 'add', 'drop']

    s = st.selectbox(f"Apply operation", options=options)

    operation = op_list[s]
    if operation is not None:
        tfm = operation.show_form(stobj=st, df=df, label=label)
        if tfm:
            try:
                df = tfm.encodes(df)
                add_operation(tfm, label)
                set_df(df)
            except BaseException as e:
                print(f'Error{e}')
