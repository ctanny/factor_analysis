'''
factor_analysis.py
Python script to perform style and factor analysis on US mutual funds.
Created by Cordell L. Tanny, CFA, FRM, FDP
February 2023
'''

import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS
from sklearn.linear_model import Lasso, LinearRegression, Ridge, ElasticNet, LassoLars
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoLarsIC
from sklearn.pipeline import make_pipeline
from dateutil.relativedelta import relativedelta
import warnings
import yfinance as yf

# prevent FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# %% initialization
start_date = '2012-01-01'
end_date = '2023-01-31'

tickers = [
    'QIACX',
    'VTSAX',
    'FCNTX',
    'TRLGX',
]

factors_dict = {
    'US Growth': 'VUG',
    'US Value': 'VTV',
    # 'US Momentum': 'MTUM',
    'US Dividend': 'SCHD',
    'US Quality': 'QUAL',
    'US Defensive': 'DEF',
    'US Small Cap': 'VB',
    'US Mid Cap': 'IJH',
    'US Credit': 'LQD',
    'US High Yield': 'JNK',
    'US Government': 'GOVT',
    'Intl Growth': 'EFG',
    'Intl Value': 'EFV',
    'Intl Momentum': 'IMTM',
    'Intl Dividend': 'IDV',
    'Intl Quality': 'IQLT',
    'Intl Small-Cap': 'GWX',
    'Intl Credit': 'IBND',
    'Intl High Yield': 'IHY',
    'Intl Government': 'IGOV',
}


# %% functions

def get_returns(tickers: list, start_date, end_date, frequency='M'):
    """
    Function to retrieve daily prices from yahoo finance
    :param tickers: list[strings]. Yahoo finance tickers supplied as a list.
    :param start_date: string.
    :param end_date: string.
    :param frequency: string. 'D' for daily, 'M' for monthly. Default is monthly.
    :return: pd.DataFrame
    """

    df_prices = yf.download(tickers, start_date, end_date)[['Adj Close']]

    # convert the index to datetime
    df_prices.index = pd.to_datetime(df_prices.index)

    # calculate the percent change based on the desired frequency
    match frequency:
        case 'D':
            df_returns = df_prices.pct_change().dropna()

        case 'M':
            df_returns = df_prices.resample('M').last().pct_change().dropna()

    # rename the columns
    df_returns.columns = tickers

    return df_returns


def retrieve_factor_returns(factors: dict, start_date: str, end_date: str, frequency='M'):
    """
    Function to retrieve factor prices from yahoo finance and return a dataframe of factor returns
    :param factors:
    :param start_date:
    :param end_date:
    :param frequency:
    :return:
    """

    df_prices = yf.download(list(factors.values()), start_date, end_date)[['Adj Close']]

    # convert the index to datetime
    df_prices.index = pd.to_datetime(df_prices.index)

    # calculate the percent change based on the desired frequency
    match frequency:
        case 'D':
            df_returns = df_prices.pct_change().dropna()

        case 'M':
            df_returns = df_prices.resample('M').last().pct_change().dropna()

    # rename the columns
    df_returns.columns = list(factors.keys())

    return df_returns
