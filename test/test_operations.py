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


def test_fill_interpolate():
    # fill float
    df = pd.DataFrame({'a': [1, np.nan, 3]})
    tfm = FillInterpolate('a')
    df_after = tfm.encodes(df)
    assert all(df_after['a'].values == [1.0, 2.0, 3.0])
    assert 'a_na' in df_after.columns

    # fill
    df = pd.DataFrame({'a': ['2022-01-02', np.nan, '2022-01-04']})
    df['a'] = pd.to_datetime(df['a'])

    df_after = tfm.encodes(df)
    assert 'a_na' in df_after.columns
    assert 'datetime' in df_after['a'].dtype.name


def test_drop_na():
    df = pd.DataFrame({'a': [1, np.nan, 3]})
    tfm = DropNA('a')

    df_after = tfm(df)
    assert len(df_after) == 2
    assert len(df.columns) == 1

    df = pd.DataFrame({'a': ['2022-01-02', np.nan, '2022-01-04']})
    df['a'] = pd.to_datetime(df['a'])

    df_after = tfm(df)
    assert len(df_after) == 2
    assert len(df.columns) == 1


def test_add_na_as_category():
    df = pd.DataFrame({'a': [1, np.nan, 3]})
    tfm = SetDType('a', 'category')
    df_after = tfm.encodes(df)
    assert len(df_after['a'].cat.categories) == 2

    tfm2 = FillNaAsCategory('a')
    df_after2 = tfm2(df_after)
    assert len(df_after2['a'].cat.categories) == 3
    cats = list(df_after2['a'].cat.categories)
    assert cats == ['NA', 1.0, 3.0]
