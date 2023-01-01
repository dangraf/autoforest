from fastcore.all import Pipeline
from autoforest.operations import *
from autoforest.clean_data import df_shrink
import pandas as pd


def get_df():
    df = pd.read_csv('test_data/Buldozers.csv', index_col=[0])
    df_test = df_shrink(df)
    return df_test


def test_cast():
    tfm_int = SetDType('SalesID', dtype='int')
    tfm_float = SetDType('SalesID', dtype='float')
    pipe = Pipeline([tfm_int, tfm_float])
    df = get_df()
    df_after = pipe(df)
    pass
