import pandas as pd
from typing import List

__all__ = ['find_cols_with_na']


def find_cols_with_na(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if df[c].isna().sum()]
