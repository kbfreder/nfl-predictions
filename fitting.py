import pandas as pd
import re
import numpy as np
import pipeline as p
import datetime as dt

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV

import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import mean_squared_error

import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy



def simple_OLS(df, x_cols, target):
    '''x_cols is list of X columns from df to use.
    Prints fit summary'''
    x_str = ' + '.join(x_cols)
    model_str = target + ' ~ ' + x_str

    y, X = patsy.dmatrices(model_str, data=df, return_type="dataframe")

    smfa_model = smf.ols(model_str, data=df)

    return smfa_model.fit()



def pick_degrees(df, x_cols, target, max_deg=5):
    '''do *not* need to include 'Ones' column in x_cols
    Plots degrees of polynomial vs train & test error '''
    X = df[x_cols]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=12)

    deg = max_deg

    train_error = np.empty(deg)
    test_error = np.empty(deg)

    for degree in range(deg):
        est = make_pipeline(PolynomialFeatures(degree), StandardScaler(), LinearRegression())
        est.fit(X_train, y_train)
        train_error[degree] = mean_squared_error(y_train, est.predict(X_train))
        test_error[degree] = mean_squared_error(y_test, est.predict(X_test))

    plt.figure(figsize=(6,4))
    plt.semilogy(np.arange(deg), train_error, color='blue', label='train')
    plt.semilogy(np.arange(deg), test_error, color='red', label='test')
    plt.ylabel('log(mean squared error)')
    plt.xlabel('degree')
    plt.legend(loc='upper left');
    plt.show();



def get_vars_low_pvals(fit, pval=0.1):
    new_vars = []
    for i in range(len(fit.pvalues)):
        if fit.pvalues[i] < pval:
            new_vars.append(fit.pvalues.index[i])
    if 'Intercept' in new_vars:
        new_vars.remove('Intercept')
    return new_vars


def assess_model_poly(df, x_cols, target, degree, print_res=True):
    '''Returns R^2 and SSE for PolynomialFeatures of degree `degree`, then Standardized & Linear Regression'''
    X = df[x_cols]
    y = df[target]

    kf = KFold(n_splits=5, random_state=19)

    model = make_pipeline(PolynomialFeatures(degree), StandardScaler(), LinearRegression())
    rs = []
    mses = []

    for train_idxs, test_idxs in kf.split(df):
        X_train = df.loc[train_idxs][x_cols]
        y_train = df.loc[train_idxs][target]
        X_test = df.loc[test_idxs][x_cols]
        y_test = df.loc[test_idxs][target]

        model.fit(X_train, y_train)

        rs.append(model.score(X_test, y_test))
        mses.append(mean_squared_error(y_test, model.predict(X_test)))
    if print_res:
        print (rs)
        print (mses)
    return [np.mean(rs), np.mean(mses)]


def assess_model_lassocv(df, x_cols, target, degree, return_coefs=False, print_res=True):
    '''Returns R^2 and SSE for PolynomialFeatures of degree `degree`, then Standardized & Linear Regression'''
    X = df[x_cols]
    y = df[target]

    kf = KFold(n_splits=5, random_state=19)

    model = make_pipeline(PolynomialFeatures(degree), StandardScaler(), LassoCV(cv=3))
    rs = []
    mses = []
    alphas = []
    coefs = []

    for train_idxs, test_idxs in kf.split(df):
        X_train = df.loc[train_idxs][x_cols]
        y_train = df.loc[train_idxs][target]
        X_test = df.loc[test_idxs][x_cols]
        y_test = df.loc[test_idxs][target]

        model.fit(X_train, y_train)

        rs.append(model.score(X_test, y_test))
        mses.append(mean_squared_error(y_test, model.predict(X_test)))
        alphas.append(model.get_params()['lassocv'].alpha_)
        coefs.append(model.get_params()['lassocv'].coef_)
    if print_res:
        print (rs)
        print (mses)
        print (alphas)
    if return_coefs:
        return [np.mean(rs), np.mean(mses), np.mean(alphas), coefs]
    else:
        return [np.mean(rs), np.mean(mses), np.mean(alphas)]


def assess_model_elasticnetcv(df, x_cols, target, degree, return_coefs=False, print_res=True):
    '''Returns R^2 and SSE for PolynomialFeatures of degree `degree`, then Standardized & Linear Regression'''
    X = df[x_cols]
    y = df[target]

    kf = KFold(n_splits=5, random_state=19)

    model = make_pipeline(PolynomialFeatures(degree), StandardScaler(), ElasticNetCV(cv=3, l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.99]))
    rs = []
    mses = []
    alphas = []
    el_ones = []
    coefs = []

    for train_idxs, test_idxs in kf.split(df):
        X_train = df.loc[train_idxs][x_cols]
        y_train = df.loc[train_idxs][target]
        X_test = df.loc[test_idxs][x_cols]
        y_test = df.loc[test_idxs][target]

        model.fit(X_train, y_train)

        rs.append(model.score(X_test, y_test))
        mses.append(mean_squared_error(y_test, model.predict(X_test)))

        alphas.append(model.get_params()['elasticnetcv'].alpha_)
        el_ones.append(model.get_params()['elasticnetcv'].l1_ratio_)
        coefs.append(model.get_params()['elasticnetcv'].coef_)

    if print_res:
        print (rs)
        print (mses)
        print (alphas)
        print (el_ones)

    if return_coefs:
        return [np.mean(rs), np.mean(mses), np.mean(alphas), np.mean(el_ones), coefs]
    else:
        return [np.mean(rs), np.mean(mses), np.mean(alphas), np.mean(el_ones)]


def assess_model_ridgecv(df, x_cols, target, degree, return_coefs=False, print_res=True):
    '''Returns R^2 and SSE for PolynomialFeatures of degree `degree`, then Standardized & Linear Regression'''
    X = df[x_cols]
    y = df[target]

    kf = KFold(n_splits=5, random_state=19)

    model = make_pipeline(PolynomialFeatures(degree), StandardScaler(), RidgeCV(cv=3))
    rs = []
    mses = []
    alphas = []
    coefs = []

    for train_idxs, test_idxs in kf.split(df):
        X_train = df.loc[train_idxs][x_cols]
        y_train = df.loc[train_idxs][target]
        X_test = df.loc[test_idxs][x_cols]
        y_test = df.loc[test_idxs][target]

        model.fit(X_train, y_train)
        rs.append(model.score(X_test, y_test))
        mses.append(mean_squared_error(y_test, model.predict(X_test)))
        alphas.append(model.get_params()['ridgecv'].alpha_)
        coefs.append(model.get_params()['ridgecv'].coef_)
        print
    if print_res:
        print (rs)
        print (mses)
        print (alphas)
    if return_coefs:
        return [np.mean(rs), np.mean(mses), np.mean(alphas), coefs]
    else:
        return [np.mean(rs), np.mean(mses), np.mean(alphas)]


def compare_models(df, x_cols, target, degree):

    # X = df[x_cols]
    # y = df[target]

    # kf = KFold(n_splits=5, random_state=19)
    rs = []
    mses = []
    alphas = []
    coefs = []

    # print('Deg 1'), assess_model_poly(df, x_cols, target, 1)
    print('Deg ' + str(degree), assess_model_poly(df, x_cols, target, degree, print_res=False))
    print('Lasso CV', assess_model_lassocv(df, x_cols, target, 1, print_res=False))
    print('Ridge CV', assess_model_ridgecv(df, x_cols, target, 1, print_res=False))
    print('Elastic Net CV', assess_model_elasticnetcv(df, x_cols, target, 1, print_res=False))
