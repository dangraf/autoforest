from fastcore.all import Transform, InplaceTransform
import pandas as pd
from autoforest.clean_data import get_na_mask

__all__ = ['Normalize',
           'FillNaAsCategory',
           'ReorderCategories',
           'SetDType']


class SetDType(InplaceTransform):
    def __init__(self, label: str, dtype: str):
        self.label = label
        self.dtype = dtype

    def setup(self, items):
        print(len(items))
        return

    def encodes(self, df: pd.DataFrame):
        df[self.label] = df[self.label].astype(self.dtype)
        return df


class Normalize(InplaceTransform):
    def __init__(self, label: str, std: float = None, mean: float = None):
        self.label = label
        self.std = std
        self.mean = mean

    def encodes(self, df: pd.DataFrame):
        if self.std is None and self.mean is None:
            self.std = df[self.label].std()
            self.mean = df[self.label].mean()
        df[self.label] = (df[self.label] - self.mean) / self.std
        return df

    def decodes(self, df: pd.DataFrame):
        df[self.label] = df[self.label] * self.std + self.mean
        return df


class FillNaAsCategory(InplaceTransform):
    def __init__(self, label):
        self.label = label

    def encodes(self, df: pd.DataFrame):
        l = list(df[self.label].cat.categories)
        if 'NA' not in l:
            l.insert(0, 'NA')
            df[self.label] = df[self.label].cat.add_categories(['NA'])
            df[self.label] = df[self.label].cat.reorder_categories(l)
            if f"{self.label}_na" in df.columns:
                df.dropna(self.label, axis='columns', inplace=True)
        df.loc[df[self.label].isna(), self.label] = 'NA'
        return df


class Fill_Median(InplaceTransform):
    def __init__(self, label: str):
        self.label = label

    def encodes(self, df):
        na_label = f"{self.label}_na"
        na = get_na_mask(df, self.label)
        df_notna = df[~na]
        idx = len(df_notna) // 2
        median = df_notna[self.label].sort_values().values[idx]
        df[na_label] = na
        df.loc[na, self.label] = median
        return df


class FillRandomSampling(InplaceTransform):
    def __init(self, label: str):
        self.label = label

    def encodes(self, df):
        na_label = f"{self.label}_na"
        na = get_na_mask(df, self.label)
        df_notna = df[~na]
        samples = df_notna[self.label].sample(n=na.sum(), replace=True)
        df[na_label] = na
        df.loc[na, self.label] = samples.values
        return df


class ReorderCategories(InplaceTransform):
    def __init__(self, label, categories):
        self.label = label
        self.categories = categories

    def encodes(self, df: pd.DataFrame):
        # todo, handle if we have fewer categories, add them
        # todo, handle if there are too many categories, how to handle that?
        df[self.label].cat.reorder_categories(self.categories)
        return df
