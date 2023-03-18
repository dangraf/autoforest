import math
import sklearn.ensemble as forest
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import export_graphviz
import IPython
import graphviz
import re

__all__ = ['rmse',
           'set_rf_samples',
           'reset_rf_samples',
           'RfRegressor',]


def rmse(x, y):
    return math.sqrt(((x - y) ** 2).mean())


def set_rf_samples(n):
    """ Changes Scikit learn's random forests to give each tree a random sample of
    n random rows.
    """
    forest._generate_sample_indices = (lambda rs, n_samples:
                                       forest.check_random_state(rs).randint(0, n_samples, n))


def reset_rf_samples():
    """ Undoes the changes produced by set_rf_samples.
    """
    forest._generate_sample_indices = (lambda rs, n_samples:
                                       forest.check_random_state(rs).randint(0, n_samples, n_samples))


class RfRegressor:
    def __init__(self, df: pd.DataFrame,
                 y_col_name,
                 time_series=False,
                 split_pct=0.2,
                 max_train_time_s: int = np.inf):
        # df: dataframe
        # y_col_name         # name of the column we are going to predict
        # split_pct = 0.2     # size of verification dataset
        # max_train_time_s = 20         # maximum runtime for one epoch of the data. If it takes longer, the data is cut during search
        # time_series = False     If set to True, the last part of the dataframe is used as validation otherwise, it's randomized

        self.df = df
        self.y_col_name = y_col_name
        self.max_train_time_s = max_train_time_s
        self.split_pct = split_pct
        self.time_series = time_series

        self.model_class = RandomForestRegressor
        self.model = None
        self.train_time = 0
        self.fit_time = 0

    def predict(self, df: pd.DataFrame):
        """
        param df: dataframe containing data as input for the model
        returns: array of predictions, array of confidence
        """
        preds = np.stack([t.predict(df) for t in self.model.estimators_])
        conf = np.std(preds, axis=0)
        pred = np.mean(preds, axis=0)
        return pred, conf

    def print_score(self, data: tuple):
        x_train, y_train, x_valid, y_valid = data
        train_preds, train_stds = self.predict(x_train)
        train_rmse = rmse(train_preds, y_train)
        train_score = self.model.score(x_train, y_train)
        text = f"train: rmse:{train_rmse:.5f} std:{train_stds.mean():.5f} score:{train_score:.5f}\n"

        valid_preds, valid_stds = self.predict(x_valid)
        valid_rmse = rmse(valid_preds, y_valid)
        valid_score = self.model.score(x_valid, y_valid)
        text = f"{text}valid: rmse:{valid_rmse:.5f} std:{valid_stds.mean():.5f} score:{valid_score:.5f}\n"
        if hasattr(self.model, 'oob_score_'):
            text = f"{text}oob_score:{self.model.oob_score:.5f}"
        print(text)
        return valid_rmse, valid_stds.mean()

    def run_rf_regressor(self, min_sample_leaf: int = 4, n_estimators: int = 40, sample_frac: float = 0.2,
                         max_features: float = 0.5):
        """
        todo, set max num samples to speed up the classification
        """
        min_sample_leaf = int(min_sample_leaf)
        n_estimators = int(n_estimators)
        max_train_time_s = np.inf if self.train_time is None else self.max_train_time_s
        data = split_data_by_time(self.df,
                                  y_col_name=self.y_col_name,
                                  split_pct=self.split_pct,
                                  time_series=self.time_series,
                                  sampled_time=self.train_time,
                                  max_train_time_s=max_train_time_s)
        t_start = time.time()
        x_train, y_train, x_valid, y_valid = data
        set_rf_samples(int(len(x_train) * sample_frac))
        self.model = self.model_class(n_estimators=n_estimators,
                                      n_jobs=-1,
                                      oob_score=True,
                                      min_samples_leaf=min_sample_leaf,
                                      max_features=max_features)

        self.model.fit(x_train, y_train)
        self.train_time = time.time() - t_start
        rmse, std = self.print_score(data)
        self.fit_time = time.time() - t_start
        return rmse, std

    def feature_importance(self):
        """ returns a dataframe containing all columns together with feature importance"""
        return pd.DataFrame(
            {'cols': self.df.drop(self.y_col_name, axis=1).columns, 'imp': self.model.feature_importances_}
        ).sort_values('imp', ascending=False)

    def plot_fi(self, df_fi: pd.DataFrame, **kwargs):
        return df_fi.plot('cols', 'imp', 'barh', legend=False, **kwargs)

    def drop_cols_low_importance(self, tresh=0.005):
        df_feat = self.model.feature_importance()
        cols_to_drop = df_feat[df_feat['imp'] < tresh]['cols']
        self.df = self.df.drop(cols_to_drop, axis=1).copy()

    def draw_tree(self, size=10, ratio=0.6, precision=0):
        """ Draws a representation of a random forest in IPython.
        Parameters:
        -----------
        t: The tree you wish to draw
        df: The data used to train the tree. This is used to get the names of the features.
        """
        t = self.model.estimators_[0]
        columns = list(self.df.columns).remove(self.y_col_name)
        s = export_graphviz(t, out_file=None, feature_names=columns, filled=True,
                            special_characters=True, rotate=True, precision=precision)
        IPython.display.display(graphviz.Source(re.sub('Tree {',
                                                       f'Tree {{ size={size}; ratio={ratio}', s)))



