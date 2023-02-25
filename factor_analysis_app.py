'''
factor_analysis_app.py
Python script to perform style and factor analysis on US mutual funds, and display as a streamlit web app.
Note that this script retrieves information from EOD Historical Data.
All users must adapt the code to his/her own data provider.
Created by Cordell L. Tanny, CFA, FRM, FDP
February 2023
'''

import streamlit as st
import os
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
import ssl
import json
from urllib.request import urlopen
import os
import plotly.express as px
import plotly.graph_objects as go

# prevent FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# context for certificates needed in urllib/requests
ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)


# remove plotly menu bar
config = {'displayModeBar': False}

# %% initialization
start_date = '2012-01-01'
end_date = '2023-01-31'

eod_api_key = os.environ['EOD_API_KEY']

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
    'US Mid Cap': 'VO',
    'US Large Cap': 'VV',
    # 'US Credit': 'LQD',
    # 'US High Yield': 'JNK',
    # 'US Government': 'GOVT',
    'Intl Growth': 'EFG',
    'Intl Value': 'EFV',
    'Intl Momentum': 'IMTM',
    'Intl Dividend': 'IDV',
    'Intl Quality': 'IQLT',
    'Intl Small-Cap': 'GWX',
    # 'Intl Credit': 'IBND',
    # 'Intl High Yield': 'IHY',
    # 'Intl Government': 'IGOV',
}

# %% functions
def get_stock_prices(symbol, start_date, end_date):
    """
    Receive the content of ``url``, parse it as JSON and return the object.
    """

    if '.TO' not in symbol:
        ticker = symbol + '.US'
    else:
        ticker = symbol

    url = f'https://eodhistoricaldata.com/api/eod/{ticker}?from={start_date}&to={end_date}&' \
          f'period=d&api_token={eod_api_key}&fmt=json'

    response = urlopen(url, context=ssl_context)
    data = response.read().decode("utf-8")

    prices = json.loads(data)

    prices = pd.DataFrame(prices).set_index('date').sort_index()

    return prices


def get_returns(tickers: list, start_date, end_date, frequency='M'):
    """
    Function to retrieve daily prices from yahoo finance
    :param tickers: list[strings]. Yahoo finance tickers supplied as a list.
    :param start_date: string.
    :param end_date: string.
    :param frequency: string. 'D' for daily, 'M' for monthly. Default is monthly.
    :return: pd.DataFrame
    """

    df_prices = pd.DataFrame()

    for ticker in tickers:
        df_temp = get_stock_prices(ticker, start_date, end_date)[['adjusted_close']]

        # convert the index to datetime
        df_temp.index = pd.to_datetime(df_temp.index)

        # add the prices to df_prices
        df_prices = pd.concat([df_prices, df_temp], axis=1)

    # calculate the percent change based on the desired frequency
    match frequency:
        case 'D':
            df_returns = df_prices.pct_change().dropna()

            # create a compound growth index

        case 'M':
            df_returns = df_prices.resample('M').last().pct_change().dropna()

            # create a compound growth index

    # rename the columns
    df_returns.columns = tickers

    print('All tickers successfully retrieved')

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

    df_prices = pd.DataFrame()

    for factor in list(factors.values()):
        df_temp = get_stock_prices(factor, start_date, end_date)[['adjusted_close']]

        # convert the index to datetime
        df_temp.index = pd.to_datetime(df_temp.index)

        # add the prices to df_prices
        df_prices = pd.concat([df_prices, df_temp], axis=1)

    # calculate the percent change based on the desired frequency
    match frequency:
        case 'D':
            df_returns = df_prices.pct_change().dropna()

        case 'M':
            df_returns = df_prices.resample('M').last().pct_change().dropna()

    # rename the columns
    df_returns.columns = list(factors.keys())

    print(f'All factors successfully retrieved')

    return df_returns


def regression_vif(X):
    # values above 5 indicate high correlation between factors
    # Calculating VIF
    X = add_constant(X)
    vif = pd.DataFrame()
    vif['variables'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return vif


def linear_reg(y, x, add_const=False):
    '''
    Calculates simple and multiple regression model using statsmodels.
    adds constant to dependent variables automatically.

    Parameters
    ----------
    y : DataFrame
        dependent variable
    x : DataFrame
        independent variables

    Returns
    -------
    statsmodels regression object

    '''
    if add_const:
        x = x.copy()

    x = add_constant(x.values)

    lm = OLS(y, x).fit()

    return lm


def multiple_lin_reg(returns, factor_returns, add_const=False):
    """
    Function to compute regular OLS regression on a group of supplied returns
    :param tickers:
    :param factors:
    :param add_const:
    :return:
    """

    df_results = pd.DataFrame()

    for ticker in returns:
        lin_reg = linear_reg(returns[ticker], factor_returns, True)
        df_results = pd.concat([df_results, lin_reg.params], axis=1)

    # drop the constant, rename columns and index
    df_results.drop('const', axis=0, inplace=True)
    df_results.index = factor_returns.columns
    df_results.columns = returns.columns
    print('Regression analysis completed')

    return df_results


def lasso_lars_regression(returns, factor_returns):
    """
    Function to calculate the regression coefficient based on LASSO
    :param tickers:
    :param factors:
    :return:
    """

    # initialize an empty dataframe to store the results
    df_results = pd.DataFrame()

    for ticker in returns:
        y = returns[ticker]
        X = factor_returns.copy()

        # run the lasso regression
        lasso_lars_ic = make_pipeline(
            StandardScaler(), LassoLarsIC(criterion="aic", normalize=False)
        ).fit(X, y)

        # select the alpha
        alpha_aic = lasso_lars_ic[-1].alpha_

        # rerun with that alpha
        best_alpha_lasso = LassoLars(alpha=alpha_aic, normalize=True)
        best_alpha_lasso.fit(X, y)
        # print(best_alpha_lasso.coef_)

        # create a df with the results
        df_temp = pd.DataFrame(data=best_alpha_lasso.coef_, index=factor_returns.columns,
                               columns=[ticker])

        df_results = pd.concat([df_results, df_temp], axis=1)

    print('Lasso regression completed')

    return df_results


# %% Streamlit code

st.header('Style and Factor Analysis Modeling')

# component 1: Add multi-select box to chose investment options
st.multiselect(
    label='Select Investments:',
    options=tickers,
)