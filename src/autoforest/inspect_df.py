import pandas as pd
import numpy as np
import scipy
from scipy.cluster import hierarchy as hc
import matplotlib.pyplot as plt
from typing import List
from statsmodels.tsa.stattools import adfuller, kpss

__all__ = ['find_cols_with_na',
           'get_adfuller_result',
           'get_kpss_result',
           'CorrelatedColumns']


def find_cols_with_na(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if df[c].isna().sum()]


def get_adfuller_result(timeseries):
    """
    https://analyticsindiamag.com/complete-guide-to-dickey-fuller-test-in-time-series-analysis/
    https://www.statsmodels.org/dev/examples/notebooks/generated/stationarity_detrending_adf_kpss.html
    """
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    return dfoutput


def get_kpss_result(timeseries):
    """
    https://analyticsindiamag.com/complete-guide-to-dickey-fuller-test-in-time-series-analysis/
    https://www.statsmodels.org/dev/examples/notebooks/generated/stationarity_detrending_adf_kpss.html
    """
    kpsstest = kpss(timeseries, regression='c')
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])
    for key, value in kpsstest[3].items():
        kpss_output['Critical Value (%s)' % key] = value
    return kpss_output


class CorrelatedColumns:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.corr = np.round(scipy.stats.spearmanr(self.df).correlation, 4)  # noqa

    def plot_var_linkage(self):
        corr_condensed = hc.distance.squareform(1 - self.corr)
        z = hc.linkage(corr_condensed, method='average')
        dendrogram = hc.dendrogram(z, labels=self.df.columns, orientation='left', leaf_font_size=16)
        plt.show(dendrogram)

    def plot_correlation_heatmap(self):
        f = plt.gcf()
        plt.matshow(self.corr, fignum=f.number)
        plt.xticks(range(self.df.shape[1]), self.df.columns, fontsize=14, rotation=90)
        plt.yticks(range(self.df.shape[1]), self.df.columns, fontsize=14)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=14)

    def _get_list_of_high_corr(self, corr_limit):
        p = np.argwhere(np.triu(np.abs(self.corr) > corr_limit, 1))
        l = list(p)
        return l

    @staticmethod
    def _del_param_in_array(arr, value):
        for i in range(len(arr) - 1, 0, -1):
            if arr[i - 1][0] == value:
                del arr[i - 1]

    def _del_duplicated_columns(self, arr):
        i = 0
        while (i < len(arr)):
            self._del_param_in_array(arr, arr[i][1])
            i += 1
        return arr

    def _arrays_to_colnames(self, arr):
        data = dict()
        for i in range(len(arr)):
            temp = data.get(arr[i][0], list())
            if len(temp) == 0:
                colname = self.df.columns[arr[i][0]]
                temp.append(colname)
            colname = self.df.columns[arr[i][1]]
            temp.append(colname)
            data[arr[i][0]] = temp
        ret = [data[k] for k in data.keys()]
        return ret

    def get_pairs(self, corr_limit: float = 0.98) -> List[List[str]]:
        """
        Return
        """
        l = self._get_list_of_high_corr(corr_limit)
        self._del_duplicated_columns(l)

        return self._arrays_to_colnames(l)

    @staticmethod
    def pairs_to_cols_to_drop(pairs: List[List[str]]) -> List[str]:
        to_drop = list()
        for p in pairs:
            to_drop.extend(p[1:])
        return to_drop
