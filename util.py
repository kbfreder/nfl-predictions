
import pandas as pd


def na_check(df):
    return df[df.isnull().any(axis=1)]


def calc_season_avg(col, df, drop=True):
    '''col (str) is column to compute Season Average (SA) for
    df (pd.DataFrame) is dataframe to merge the result back into
    drop (Boolean): whether or not to drop the original column
    '''
    base_cols = ['Team','Season','Date']
    mini_df = df[base_cols + [col]]
    df2 = pd.merge(mini_df[base_cols], mini_df, on=['Team','Season'])
    df3 = df2[df2['Date_x'] > df2['Date_y']]
    df4 = df3.groupby(['Team', 'Season', 'Date_x',]).mean().reset_index()
    df4.rename(columns={'Date_x':'Date', col: col + '_SA'}, inplace=True)
    df4.drop(columns='Season', inplace=True)

    df = pd.merge(df, df4, how='left', on=['Team','Date'])

    if drop:
        df.drop(columns=col, inplace=True)

    return df
