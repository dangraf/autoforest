import pickle
from pathlib import Path

import numpy as np
from typing import List
from enum import Enum
import re
from numpy import int8, int16, int32, int64
from numpy import uint8, uint16, uint32, uint64
from numpy import float32, float64, longdouble

__all__ = ['cast_val_to_dtype',
           'df_shrink',
           'is_cat',
           'cont_cat_split',
           'get_na_mask',
           'get_inf_mask',
           'fill_median',
           'fill_random_sampling',
           'NormalizedDtype',
           'apply_operations_file',
           'add_datepart',
           'split_by_sorded_column']

import pandas as pd


def cast_val_to_dtype(dtype, value):
    return eval(dtype.name)(value)


class NormalizedDtype(Enum):
    Int = 0
    Float = 1
    Categorical = 2
    Datetime = 3

    @staticmethod
    def get_list_of_types():
        l = [v.name for v in NormalizedDtype]
        return [' '] + l

    @staticmethod
    def get_index_from_dtype(dtype):
        if 'int' in dtype.name.lower():
            return NormalizedDtype.Int.value
        elif 'float' in dtype.name.lower():
            return NormalizedDtype.Float.value
        elif 'catego' in dtype.name.lower():
            return NormalizedDtype.Categorical.value
        else:
            return NormalizedDtype.Datetime.value

    @staticmethod
    def get_normalized_dtype(dtype):
        if 'int' in dtype.name.lower():
            return NormalizedDtype.Int
        elif 'float' in dtype.name.lower():
            return NormalizedDtype.Float
        elif 'catego' in dtype.name.lower():
            return NormalizedDtype.Categorical
        elif 'object' in dtype.name.lower():
            return NormalizedDtype.Categorical
        else:
            return NormalizedDtype.Datetime

    @staticmethod
    def try_convert(column: pd.Series, stype: str):
        try:
            if stype == NormalizedDtype.Int.name:
                s = column.astype('int')
            elif stype == NormalizedDtype.Float.name:
                s = column.astype('float')
            elif stype == NormalizedDtype.Categorical:
                s = column.astype('category')
            else:
                column = column.astype('object')
                s = pd.to_datetime(column)
        except BaseException as e:
            s = column

        return s

        pass


def is_cat(df, label, max_card=20):
    if ((pd.api.types.is_integer_dtype(df[label].dtype) and
         df[label].unique().shape[0] > max_card) or
            pd.api.types.is_float_dtype(df[label].dtype)):
        return False
    else:
        return True


def get_na_mask(df, label):
    na_label = f"{label}_na"
    if na_label in df.columns:
        na = df[na_label]
    else:
        na = df[label].isna()
    return na


def get_inf_mask(df, label):
    inf_label = f"{label}_inf"
    if inf_label in df.columns:
        inf = df[inf_label]
    else:
        inf = np.isinf(df[label])
    return inf


def fill_random_sampling(df: pd.DataFrame, label):
    na_label = f"{label}_na"
    na = get_na_mask(df, label)
    df_notna = df[~na]
    samples = df_notna[label].sample(n=na.sum(), replace=True)
    df[na_label] = na
    df.loc[na, label] = samples.values


def fill_median(df: pd.DataFrame, label: str):
    na_label = f"{label}_na"
    na = get_na_mask(df, label)
    df_notna = df[~na]
    idx = len(df_notna) // 2
    median = df_notna[label].sort_values().values[idx]
    df[na_label] = na
    df.loc[na, label] = median


def cont_cat_split(df, max_card=20, dep_var=None):
    "Helper function that returns column names of cont and cat variables from given `df`."
    cont_names, cat_names = [], []
    for label in df:
        if label in list(dep_var):
            continue
        if ((pd.api.types.is_integer_dtype(df[label].dtype) and
             df[label].unique().shape[0] > max_card) or
                pd.api.types.is_float_dtype(df[label].dtype)):
            cont_names.append(label)
        else:
            cat_names.append(label)
    return cont_names, cat_names


def df_shrink_dtypes(df, skip: List[str] = [], obj2cat=True, int2uint=False):
    "Return any possible smaller data types for DataFrame columns. Allows `object`->`category`, `int`->`uint`, and exclusion."

    # 1: Build column filter and typemap
    excl_types, skip = {'category', 'datetime64[ns]', 'bool'}, set(skip)

    typemap = {
        'int': [(np.dtype(x), np.iinfo(x).min, np.iinfo(x).max) for x in (np.int8, np.int16, np.int32, np.int64)],
        'uint': [(np.dtype(x), np.iinfo(x).min, np.iinfo(x).max) for x in (np.uint8, np.uint16, np.uint32, np.uint64)],
        'float': [(np.dtype(x), np.finfo(x).min, np.finfo(x).max) for x in (np.float32, np.float64, np.longdouble)]
    }
    if obj2cat:
        typemap['object'] = 'category'  # User wants to categorify dtype('Object'), which may not always save space
    else:
        excl_types.add('object')

    new_dtypes = {}
    exclude = lambda dt: dt[1].name not in excl_types and dt[0] not in skip

    for c, old_t in filter(exclude, df.dtypes.items()):
        t = next((v for k, v in typemap.items() if old_t.name.startswith(k)), None)

        if isinstance(t, list):  # Find the smallest type that fits
            if int2uint and t == typemap['int'] and df[c].min() >= 0: t = typemap['uint']
            new_t = next((r[0] for r in t if r[1] <= df[c].min() and r[2] >= df[c].max()), None)
            if new_t and new_t == old_t: new_t = None
        else:
            new_t = t if isinstance(t, str) else None

        if new_t:
            new_dtypes[c] = new_t
    return new_dtypes


def categories_to_str(df: pd.DataFrame, skip: List[str] = []):
    """
    Converts all categorical values to strings since mixed types for categories
    throws errors while plotting and saving dataframes
    """

    def exclude(dt):
        return dt[0] not in skip

    for label, dtype in filter(exclude, df.dtypes.items()):
        if dtype.name == 'category':
            df[label] = df[label].apply(lambda x: str(x))
            df[label] = df[label].astype('category')


def df_shrink(df, skip: List[str] = [], obj2cat=True, int2uint=False) -> pd.DataFrame:
    "Reduce DataFrame memory usage, by casting to smaller types returned by `df_shrink_dtypes()`."
    dt = df_shrink_dtypes(df, skip, obj2cat=obj2cat, int2uint=int2uint)
    df = df.astype(dt)
    categories_to_str(df, skip=skip)
    return df


def make_date(df, date_field):
    "Make sure `df[date_field]` is of the right date type."
    field_dtype = df[date_field].dtype
    if isinstance(field_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        field_dtype = np.datetime64
    if not np.issubdtype(field_dtype, np.datetime64):
        df[date_field] = pd.to_datetime(df[date_field], infer_datetime_format=True)


def add_datepart(df, field_name, prefix=None, drop=True, time=False):
    "Helper function that adds columns relevant to a date in the column `field_name` of `df`."
    make_date(df, field_name)
    field = df[field_name]
    if prefix is None:
        prefix = re.sub('[Dd]ate$', '', field_name)
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear', 'Is_month_end', 'Is_month_start',
            'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    # Pandas removed `dt.week` in v1.1.10
    week = field.dt.isocalendar().week.astype(field.dt.day.dtype) if hasattr(field.dt, 'isocalendar') else field.dt.week
    for n in attr: df[prefix + n] = getattr(field.dt, n.lower()) if n != 'Week' else week
    mask = ~field.isna()
    df[prefix + 'Elapsed'] = np.where(mask, field.values.astype(np.int64) // 10 ** 9, np.nan)
    if drop: df.drop(field_name, axis=1, inplace=True)
    return df


def apply_operations_file(df, path_to_operations='/home/user/Downloads/dataframe.pcl'):
    with open(path_to_operations, 'rb') as f:
        ops = pickle.load(f)
    for item in ops.items():
        col = item[0]
        for op in item[1]:
            df = op(df)


def read_dataframe(uploaded_file):
    """
    reads the dataframe and tries to convert to correct dtypes and shrinks it to save memory
    """
    p = Path(uploaded_file.name)
    print(p.suffix)
    if p.suffix == '.csv':
        df = pd.read_csv(uploaded_file, index_col=[0])
    elif p.suffix == '.json':
        df = pd.read_json(uploaded_file)
    elif p.suffix == '.pcl':
        df = pd.read_picke(uploaded_file)
    elif p.suffix == '.feather':
        df = pd.read_feather(uploaded_file)
    else:
        df = pd.DataFrame()
    return df_shrink(df)


def split_data(df_x, df_y, pct):
    "Splits the data into train and validation set"
    " pct: 0-1 depending on how much that is in validation set"
    df_x.reset_index(inplace=True, drop=True)
    df_y.reset_index(inplace=True, drop=True)

    l = int(len(df_x) * pct)
    x_train = df_x.iloc[:-l]
    x_valid = df_x.iloc[-l:]
    y_train = df_y.iloc[:-l]
    y_valid = df_y.iloc[-l:]
    return x_train, y_train, x_valid, y_valid


def split_by_sorded_column(df, dep_var, pct=0.2, ascending=False):
    l = len(df)
    split_index = int(l * pct)
    valid = df.sort_values('year', ascending=ascending).iloc[:split_index]
    train = df.sort_values('year', ascending=ascending).iloc[split_index:]

    valid = valid.sample(frac=1.0)
    train = train.sample(frac=1.0)

    train_x = train.drop(dep_var, axis=1)
    train_y = train[dep_var]
    valid_x = valid.drop(dep_var, axis=1)
    valid_y = valid[dep_var]
    return train_x, train_y, valid_x, valid_y


