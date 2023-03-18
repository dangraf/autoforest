from fastcore.all import Transform, InplaceTransform
import pandas as pd
# same types as df_shring_dtypes to be able to cast result to correct value
from numpy import int8, int16, int32, int64
from numpy import uint8, uint16, uint32, uint64
from numpy import float32, float64, longdouble
import numpy as np
import streamlit as st

from autoforest.prepare_data import get_na_mask, cast_val_to_dtype, add_datepart

MAX_LEN_REORDER = 25

__all__ = ['BaseTransform',
           'TfmDiff',
           'TfmNormalize',
           'TfmReplace',
           'TfmExp',
           'TfmLog',
           'TfmAdd',
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
           'DropNA',
           'DropCol',
           'TfmAddDatePart']


class BaseTransform(InplaceTransform):
    def __init__(self, label):
        super().__init__()
        self.label = label

    def __repr__(self):
        return f"{self.name}"

    @classmethod
    def show_form(cls, stobj: st, df: pd.DataFrame, label: str):
        if stobj.button(f"apply {cls.__name__}"):
            return cls(label)


class SetDType(BaseTransform):
    def __init__(self, label: str, dtype: str):
        super().__init__(label)
        self.label = label
        self.dtype = dtype.lower()

    def encodes(self, df: pd.DataFrame):
        print(f"setting dtype {self.dtype}")
        if self.dtype == 'datetime':
            df[self.label]  = df[self.label].astype('object')
            df[self.label] = pd.to_datetime(df[self.label], infer_datetime_format=True)
            print(f"dtype: {df[self.label].dtype}")
        else:
            df[self.label] = df[self.label].astype(self.dtype)
        return df

    def __repr__(self):
        return f"{self.name} {self.dtype}"


class TfmAdd(BaseTransform):
    def __init__(self, label, const):
        super().__init__(label)
        self.const = const

    def encodes(self, df: pd.DataFrame):
        const = cast_val_to_dtype(df[self.label].dtype, self.const)
        df[self.label] += const
        return df

    def decodes(self, df: pd.DataFrame):
        const = cast_val_to_dtype(df[self.label].dtype, self.const)
        df[self.label] -= const
        return df

    def __repr__(self):
        return f"{self.name} {self.const}"

    @classmethod
    def show_form(cls, stobj: st, df: pd.DataFrame, label: str):
        with stobj.form("replace value", clear_on_submit=True):
            const = stobj.text_input('value to add:')
            submitted = stobj.form_submit_button("add value")
            if submitted:
                return TfmAdd(label=label, const=const)


class TfmReplace(BaseTransform):
    def __init__(self, label, target_val, new_val):
        super().__init__(label)
        self.tval = int(target_val)
        self.rval = int(new_val)

    def encodes(self, df: pd.DataFrame):
        filt = df[self.label] == self.tval
        df.loc[filt, self.label] = self.rval
        return df

    def __repr__(self):
        return f"{self.name} {self.tval}>>{self.rval}"

    @classmethod
    def show_form(cls, stobj: st, df: pd.DataFrame, label: str):
        """
        returns tfmReplace-object
        """
        with stobj.form("replace value", clear_on_submit=True):
            target = stobj.text_input('Replace value:')
            new_value = stobj.text_input('with:')
            submitted = stobj.form_submit_button("Replace")
            if submitted:
                return TfmReplace(label=label, target_val=target, new_val=new_value)


class TfmDiff(BaseTransform):
    def __init__(self, label, steps=1):
        super().__init__(label)
        self.steps = steps

    def encodes(self, df):
        df[self.label].diff(self.steps)
        return df

    def __repr__(self):
        return f"{self.name}({self.steps})"

    @classmethod
    def show_form(cls, stobj: st, df: pd.DataFrame, label: str):
        target = stobj.text_input('diff steps(int)')
        if stobj.button('apply'):
            steps = int(target)
            return TfmDiff(label, steps=steps)


class TfmExp(BaseTransform):

    def encodes(self, df: pd.DataFrame):
        df[self.label] = np.exp(df[self.label])
        return df

    def decodes(self, df: pd.DataFrame):
        df[self.label] = np.log(df[self.label])
        return df


class TfmLog(BaseTransform):
    def __init__(self, label, epsilon=1e-3):
        super().__init__(label)
        self.epsilon = epsilon

    def encodes(self, df: pd.DataFrame):
        df[self.label] = np.log(df[self.label] + self.epsilon)
        return df

    def decodes(self, df: pd.DataFrame):
        df[self.label] = np.exp(df[self.label]) - self.epsilon
        return df

    def __repr__(self):
        return f"{self.name}, epsilon:{self.epsilon}"


class TfmNormalize(BaseTransform):
    def __init__(self, label: str, std: float = None, mean: float = None):
        super().__init__(label)
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


class FillNaAsCategory(BaseTransform):

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


class FillMedian(BaseTransform):
    def __init__(self, label: str):
        super().__init__(label)
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


class FillMean(BaseTransform):
    def __init__(self, label: str):
        super().__init__(label)
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


class FillRandomSampling(BaseTransform):

    # todo, add values in object to sample from
    def __init__(self, label, all_values: pd.Series = None):
        super().__init__(label)
        self.all_values = all_values

    def encodes(self, df):
        _add_na_column(df, self.label)
        na = get_na_mask(df, self.label)
        df_notna = df[~na]
        if self.all_values is None:
            self.all_values = df_notna[self.label]
        samples = self.all_values.sample(n=na.sum(), replace=True)
        df.loc[na, self.label] = samples.values
        return df

    def __repr__(self):
        return f"{self.name}"


class ReorderCategories(BaseTransform):
    def __init__(self, label, categories):
        super().__init__(label)
        self.categories = categories

    def encodes(self, df: pd.DataFrame):
        # todo, handle if we have fewer categories, add them
        # todo, handle if there are too many categories, how to handle that?
        df[self.label] = df[self.label].cat.set_categories(new_categories=self.categories, ordered=True)
        df[self.label] = df[self.label].cat.reorder_categories(self.categories, ordered=True)
        print(df[self.label].cat.categories)
        return df

    def __repr__(self):
        return f"{self.name} {self.categories}"

    @classmethod
    def show_form(cls, stobj, df, label):
        num_cats = len(df[label].cat.categories)
        options = list(range(num_cats))
        selections = list()
        categories = list(df[label].cat.categories)
        if len(categories) > MAX_LEN_REORDER:
            # datetimes etc can be categorical but seldom need to be ordered
            return
        cats = ', '.join(list(df[label].cat.categories))
        stobj.write(f"**Current order of categories:** \n {cats}")
        for i, cat in enumerate(categories):
            s = stobj.selectbox(f"{cat}", options=options, index=i)
            selections.append(s)
        for i, sel in enumerate(selections):
            if i != sel:
                categories[sel], categories[i] = categories[i], categories[sel]
                df[label] = df[label].cat.reorder_categories(categories, ordered=True)
                stobj.experimental_rerun()
                break
        if stobj.button('Apply Reorder'):
            return ReorderCategories(label=label, categories=categories)


class FillConstant(BaseTransform):
    def __init__(self, label, constant):
        super().__init__(label)
        self.constant = constant

    def encodes(self, df: pd.DataFrame):
        self.constant = cast_val_to_dtype(df[self.label].dtype, self.constant)
        _add_na_column(df, self.label)
        na_mask = get_na_mask(df, self.label)
        df.loc[na_mask, self.label] = self.constant
        return df

    def __repr__(self):
        return f"{self.name} {self.constant}"

    @classmethod
    def show_form(cls, stobj: st, df: pd.DataFrame, label: str):
        with stobj.form("Fill Constant Value", clear_on_submit=True):
            const = stobj.text_input('Value:')
            submitted = stobj.form_submit_button("Replace")
            if submitted:
                return FillConstant(label=label, constant=const)


class FillFwd(BaseTransform):

    def encodes(self, df: pd.DataFrame):
        _add_na_column(df, self.label)

        df[self.label].ffill(inplace=True)
        return df

    def __repr__(self):
        return f"{self.name}"


class FillBwd(BaseTransform):

    def encodes(self, df: pd.DataFrame):
        _add_na_column(df, self.label)

        df[self.label].bfill(inplace=True)
        return df

    def __repr__(self):
        return f"{self.name}"


class FillInterpolate(BaseTransform):
    def __init__(self, label, **kwargs):
        super().__init__(label)
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


class DropNA(BaseTransform):

    def encodes(self, df: pd.DataFrame):
        df.dropna(subset=[self.label], axis='index', inplace=True)
        return df

    def __repr__(self):
        return f"{self.name}"


class DropCol(BaseTransform):
    def encodes(self, df: pd.DataFrame):
        print(self.label)
        df.drop(self.label, axis=1, inplace=True)
        return df

    # @classmethod
    # def show_form(cls, stobj: st, df: pd.DataFrame, label: str):
    #    if stobj.button('add drop transform'):
    #        return DropCol(label)

    def __repr__(self):
        return f"{self.name}"


class TfmAddDatePart(BaseTransform):
    def encodes(self, df: pd.DataFrame):
        df = add_datepart(df, self.label)
        return df
