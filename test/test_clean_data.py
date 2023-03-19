
import pandas as pd

from autoforest.prepare_data import df_shrink


def test_shringk_data():
    df = pd.read_csv('test_data/Buldozers.csv', index_col=[0])
    df_test = df_shrink(df)
    pass
