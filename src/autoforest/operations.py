import pandas as pd

def op_set_type(data:pd.Series, typ)->pd.Series:
    return data.astype(typ)

def op_normalize(data:pd.Series, mean, std)->pd.Series:
    return (data-mean)/std
