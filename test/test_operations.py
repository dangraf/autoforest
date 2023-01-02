from fastcore.all import Pipeline
from autoforest.operations import *
from autoforest.clean_data import df_shrink
import pandas as pd
import numpy as np


def get_df():
    df = pd.read_csv('test_data/Buldozers.csv', index_col=[0])
    df_test = df_shrink(df)
    return df_test


def test_cast():
    tfm_int = SetDType('a', dtype='int')
    tfm_int.order = 0
    tfm_float = SetDType('a', dtype='float')
    tfm_float.order = 1
    pipe = Pipeline([tfm_int, tfm_float])
    df = pd.DataFrame({'a': [1, 2, 3]})
    df_after = pipe(df)
    assert all(df_after['a'].values == [1.0, 2.0, 3.0])

    operation = SetDType
    kwargs = {'label': 'a', 'dtype': 'int'}
    tfm = operation(**kwargs)
    df_after = tfm.encodes(df)
    assert all(df_after['a'].values == [1, 2, 3])

def test_fill_constant():
    pass
