import pickle
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from selenium.common.exceptions import NoSuchElementException
from collections import defaultdict
import csv

url_base = 'https://www.pro-football-reference.com'
week_conv_dict = {'Wild Card':18, 'Division':19, 'Conf. Champ.':20, 'SuperBowl':21}


def save_dict_txt(filename, d):
    with open(filename,"w") as f:
        f.write(str(d))
        f.close()


def init_scraping():
    bxsc_hmtl = {}    # [href] = html .also tracks whether a bxsc href has been visited
    bxsc_refs = {}      # [team-year] = list of hrefs for boxscores
    season_dict = defaultdict(list)     # [year] = list of teams whose team-season page has been sucessfully scraped
    season_html = {}    # [team-year] = html of team season page
    # bxsc_html = defaultdict(list)       # [team-year] = list of html for boxscores (that )
    # bxsc_dict = defaultdict(list)


def init_cleaning():
    master_df = pd.DataFrame()      # master dataframe containing all data
    tsdf_merge_track = defaultdict(list)  # [year] = list of teams whose df has been merged into master df


def load_things():
    href_dict = open_pkl('href_dict.pkl')
    # bxsc_df = pd.read_csv('bxsc_df.csv')
    master_df = open_pkl('master_df.pkl')
    merge_dict = open_pkl('merge_dict.pkl')
    season_dict = open_pkl('season_dict.pkl')
    season_html = open_pkl('season_html.pkl')
    # return href_dict, bxsc_df, master_df, merge_dict, season_dict, season_html
    return href_dict, master_df, merge_dict, season_dict, season_html

def pkl_this(filename, df):
    '''Saves df as filename. Must include .pkl extension'''
    with open(filename, 'wb') as picklefile:
        pickle.dump(df, picklefile)

def open_pkl(filename):
    '''Must include .pkl extension. Returns object that can be saved to a df.
    '''
    with open(filename,'rb') as picklefile:
        return pickle.load(picklefile)

def log_this(s):
    with open("log.txt", "a") as f:
        f.write("\n" + s + "\n")

def close_dr():
    try:
        driver.close()
        print('Driver closed')
    except:
        print('Already closed')

def row_split(row,i):
    return int(row.split('-')[i])


def get_season_data(team_url, year, count, driver):
    '''team_url (str): team abbreviation used in PFR url (ex: 'chi')
    year (int): season year
    count (int): count of number of times driver has been called
    driver: Selenium WebDriver
    '''
    url = url_base + '/teams/' + team_url + '/' + str(year) + '.htm'

    print('Loading page for ' + team_url, year)

    try:
        driver.get(url)
        # time.sleep(3 + count//2)
        time.sleep(3)
        target_id = "all_games"

        try:
            webel = driver.find_element_by_id(target_id)
            print('Page loaded & data found!')

            html = webel.get_attribute('outerHTML')
            fname = 'SeasonHtmls/' + team_url + '-' + str(year) + '-html.pkl'
            pkl_this(fname, html)
            return html

        except NoSuchElementException:
            print ('Cannot find element id. Check page')
            log_this('Error - cannot find elelment id for ' + team_url + '-' + str(year))
            return None

    except TypeError:
        print ('Error loading page.')
        log_this('Error loading page for ' + team_url + '-' + str(year))
        return None


def get_href_list(html):
    soup = BeautifulSoup(html, 'lxml')
    r2 = re.compile(r'(boxscores)')
    href_list = soup.find_all('a', href=r2)
    return href_list


def clean_season_data(html, team, year):
    df = pd.read_html(html)[0]

    if len(df.columns) < 10:
        df = pd.read_html(html)[1]

    col_list_multi = list(df.columns)
    col_list_flat = [(c[0] + ' ' + c[1]).strip('Unnamed: ').strip() for c in col_list_multi]

    r1 = re.compile(r'([0-9]+_level_0 )')
    col_list_flat = [re.sub(r1,'',s) for s in col_list_flat]

    cols1 = ['Week','Day','Date','Time','bs','Result','OT','Record','Location','Opp','PtsTm','PtsOpp']
    col_list_final = cols1 + col_list_flat[-(len(cols1)+1):]

    # cols2 = ['OExpPts','DExpPts', 'StExpPts']
    # col_list_final = col_list_final[:-len(cols2)] + cols2

    # labe the first '1stD' column to distinguish it and allow us to find the second set of these columns
    st = col_list_final.index('1stD')
    col_list_final[st] += '-O'

    st = col_list_final.index('1stD')
    for i in range(5):
        old_str = col_list_final[st+i]
        col_list_final[st+i] += '-D'

    df.columns = col_list_final
    df['Team'] = team
    df['Year'] = year
    cols_to_drop = ['1stD-D', 'TotY-D', 'PassY-D', 'RushY-D', 'Offens', 'Defens', 'Sp. Tms']
    df.drop(cols_to_drop, axis=1, inplace=True)
    # df.rename(columns={'1stDO-O':'1stD','TO-O': 'TO', 'TO-D':'DefTO'},inplace=True)   # can do this later

    # convert week to integers, drop na's
    # Week = na when it's the Playoffs and team didn't make it
    # bs = na when team is on Bye Week
    df.dropna(axis=0, subset=['Week','bs'], inplace=True)
    df['Week'] = df['Week'].apply(lambda x: week_conv_dict[x] if x in week_conv_dict else int(x))

    fname = '../SeasonDfs/season_' + team + '-' + str(year) + '.pkl'
    pkl_this(fname, df)

    return df


def get_bxsc_data(href, driver):
    '''href (str): url fragment for boxscore
       week (int): week of season
       driver (Selenium WebDriver)
    '''
    url2 = url_base + href
    # print('Loading boxscore page')
    driver.get(url2)
    time.sleep(3)
    try:
        webel = driver.find_element_by_id('team_stats')
        # print('Page loaded & Team Stats found!')
        html = webel.get_attribute('outerHTML')
        return html
    except NoSuchElementException:
        print('Skipping page ' + url2)



def clean_bxsc_data(html,week, year):
    hlist = pd.read_html(html)
    df3 = hlist[0]
    df3.rename(columns={'Unnamed: 0':'Stats'}, inplace=True)
    df3.set_index('Stats', inplace=True)
    df3 = df3.transpose()

    cols_to_drop = ['First Downs','Total Yards','Turnovers']
    df3.drop(cols_to_drop, axis=1, inplace=True)

    clist = list(df3.columns)
    cols_to_drop2 = clist[:8]

    bxsc_trns = [['RushAtt', clist[0], 0],
                 ['RushTDs', clist[0], 2],
                 ['PassCmp', clist[1], 0],
                 ['PassAtt', clist[1], 1],
                 ['PassTDs', clist[1], -2],
                 ['INT', clist[1], -1],
                 ['SacksO', clist[2], 0],
                 ['Fumbles', clist[4], 0],
                 ['Penalies', clist[5], 0],
                 ['PenY', clist[5], 1],
                 ['3rdDConv', clist[6], 0],
                 ['3rdDAtt', clist[6], 1],
                 ['4thDAtt', clist[7], 1],
                ]

    for i in bxsc_trns:
        df3[i[0]] = df3[i[1]].apply(lambda row: row_split(row,i[2]))

    df3['Week'] = week
    df3['Year'] = year

    df3.drop(cols_to_drop2, axis=1, inplace=True)
    df3.reset_index(inplace=True)
    df3.rename(columns={'index':'Team'}, inplace=True)

    return df3


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
