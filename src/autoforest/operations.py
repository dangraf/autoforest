from fastcore.all import Transform, InplaceTransform
import pandas as pd
# same types as df_shring_dtypes to be able to cast result to correct value
from numpy import int8, int16, int32, int64
from numpy import uint8, uint16, uint32, uint64
from numpy import float32, float64, longdouble
import numpy as np

from autoforest.clean_data import get_na_mask

__all__ = ['Normalize',
           'ReorderCategories',
           'SetDType',
           'FillNaAsCategory',
           'FillMedian',
           'FillRandomSampling',
           'FillFwd',
           'FillBwd',
           'FillMean',
           'FillInterpolate',
           'FillConstant',
           'DropNA', ]


class SetDType(InplaceTransform):
    def __init__(self, label: str, dtype: str):
        super().__init__()
        self.label = label
        self.dtype = dtype.lower()

    def encodes(self, df: pd.DataFrame):
        df[self.label] = df[self.label].astype(self.dtype)
        return df

    def __repr__(self):
        return f"{self.name} {self.dtype}"


class Normalize(InplaceTransform):
    def __init__(self, label: str, std: float = None, mean: float = None):
        super().__init__()
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

    def __repr__(self):
        return f"{self.name} std: {self.std:.2f} mean: {self.mean:.2f}"


def _add_na_column(df, label):
    na_label = f"{label}_na"
    if na_label not in df.columns:
        na = get_na_mask(df, label)
        df[na_label] = na


class FillNaAsCategory(InplaceTransform):
    def __init__(self, label):
        super().__init__()
        self.label = label

    def encodes(self, df: pd.DataFrame):
        cats = list(df[self.label].cat.categories)
        if 'NA' not in cats:
            cats.insert(0, 'NA')
            df[self.label] = df[self.label].cat.add_categories(['NA'])
            df[self.label] = df[self.label].cat.reorder_categories(cats)
            if f"{self.label}_na" in df.columns:
                df.dropna(subset=[self.label], axis='columns', inplace=True)
        df.loc[df[self.label].isna(), self.label] = 'NA'
        return df


class FillMedian(InplaceTransform):
    def __init__(self, label: str):
        super().__init__()
        self.label = label
        self.median = None

    def encodes(self, df):
        _add_na_column(df, self.label)
        na = get_na_mask(df, self.label)
        df_notna = df[~na]
        if self.median is None:
            idx = len(df_notna) // 2
            median = df_notna[self.label].sort_values().values[idx]
            self.median = median
        df.loc[na, self.label] = self.median
        return df

    def __repr__(self):
        return f"{self.name} {self.median}"


class FillMean(InplaceTransform):
    def __init__(self, label: str):
        super().__init__()
        self.label = label
        self.mean = None

    def encodes(self, df):
        _add_na_column(df, self.label)
        na = get_na_mask(df, self.label)
        df_notna = df[~na]
        if self.mean is None:
            self.mean = df_notna[self.label].mean()
        df.loc[na, self.label] = self.mean
        return df

    def __repr__(self):
        return f"{self.name} {self.mean:.2f}"


class FillRandomSampling(InplaceTransform):
    def __init__(self, label):
        super().__init__()
        self.label = label

    def encodes(self, df):
        _add_na_column(df, self.label)
        na = get_na_mask(df, self.label)
        df_notna = df[~na]
        samples = df_notna[self.label].sample(n=na.sum(), replace=True)
        df.loc[na, self.label] = samples.values
        return df

    def __repr__(self):
        return f"{self.name}"


class ReorderCategories(InplaceTransform):
    def __init__(self, label, categories):
        super().__init__()
        self.label = label
        self.categories = categories

    def encodes(self, df: pd.DataFrame):
        # todo, handle if we have fewer categories, add them
        # todo, handle if there are too many categories, how to handle that?
        df[self.label].cat.reorder_categories(self.categories)
        return df

    def __repr__(self):
        return f"{self.name} {self.categories}"


class FillConstant(InplaceTransform):
    def __init__(self, label, constant):
        super().__init__()
        self.label = label
        self.constant = constant

    def encodes(self, df: pd.DataFrame):
        self.constant = eval(df[self.label].dtype.name)(self.constant)
        _add_na_column(df, self.label)
        na_mask = get_na_mask(df, self.label)
        df.loc[na_mask, self.label] = self.constant
        return df

    def __repr__(self):
        return f"{self.name} {self.constant}"


class FillFwd(InplaceTransform):
    def __init__(self, label):
        super().__init__()
        self.label = label

    def encodes(self, df: pd.DataFrame):
        _add_na_column(df, self.label)

        df[self.label].ffill(inplace=True)
        return df

    def __repr__(self):
        return f"{self.name}"


class FillBwd(InplaceTransform):
    def __init__(self, label):
        super().__init__()
        self.label = label

    def encodes(self, df: pd.DataFrame):
        _add_na_column(df, self.label)

        df[self.label].bfill(inplace=True)
        return df

    def __repr__(self):
        return f"{self.name}"


class FillInterpolate(InplaceTransform):
    def __init__(self, label, **kwargs):
        super().__init__()
        self.label = label
        self.method = 'linear'
        self.kwargs = kwargs

    def encodes(self, df: pd.DataFrame):
        _add_na_column(df, self.label)
        conv_to_datetime = False
        if 'datetime' in df[self.label].dtype.name:
            conv_to_datetime = True
            df[self.label] = df[self.label].astype(int64)
            df[self.label][df[self.label] < 0] = np.nan

        df[self.label].interpolate(method=self.method, inplace=True, **self.kwargs)

        if conv_to_datetime:
            df[self.label] = pd.to_datetime(df[self.label], unit='ns')
        return df

    def __repr__(self):
        return f"{self.name} {self.method}, {self.kwargs}"


class DropNA(InplaceTransform):
    def __init__(self, label):
        super().__init__()
        self.label = label

    def encodes(self, df: pd.DataFrame):
        df.dropna(subset=[self.label], axis='index', inplace=True)
        return df

    def __repr__(self):
        return f"{self.name}"
