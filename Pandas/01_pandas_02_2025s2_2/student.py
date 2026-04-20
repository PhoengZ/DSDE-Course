import pandas as pd
import json

"""
    ASSIGNMENT 1 (STUDENT VERSION):
    Using pandas to explore youtube trending data from (videos.csv and category_id.json) and answer the questions.
"""

def Q1():
    """
        1. How many rows are there in the videos.csv after removing duplications?
        - To access 'videos.csv', use the path '/data/videos.csv'.
    """
    df = pd.read_csv("/data/videos.csv").drop_duplicates()
    return df.shape[0]

def Q2(vdo_df):
    '''
        2. How many VDO that have "dislikes" more than "likes"? Make sure that you count only unique title!
            - videos.csv has been loaded into memory and is ready to be utilized as vdo_df
            - The duplicate rows of vdo_df have been removed.
    '''
    return vdo_df.loc[vdo_df['dislikes'] > vdo_df['likes'], 'title'].nunique()

def Q3(vdo_df):
    '''
        3. How many VDO that are trending on 22 Jan 2018 with comments more than 10,000 comments?
            - videos.csv has been loaded into memory and is ready to be utilized as vdo_df
            - The duplicate rows of vdo_df have been removed.
            - The trending date of vdo_df is represented as 'YY.DD.MM'. For example, January 22, 2018, is represented as '18.22.01'.
    '''
    vdo_df['trending_date'] = pd.to_datetime(vdo_df['trending_date'], format="%y.%d.%m")
    return vdo_df.loc[(vdo_df['trending_date'] == pd.to_datetime("18.22.1", format="%y.%d.%m"))&(vdo_df['comment_count'] > 10000), :].shape[0]

def Q4(vdo_df):
    '''
        4. Which trending date that has the minimum average number of comments per VDO?
            - videos.csv has been loaded into memory and is ready to be utilized as vdo_df
            - The duplicate rows of vdo_df have been removed.
    '''
    vdo_df['trending_date'] = pd.to_datetime(vdo_df['trending_date'], format="%y.%d.%m")
    groupby_vdo = vdo_df.groupby(['trending_date'])['comment_count'].mean()
    mv = groupby_vdo.min()
    return groupby_vdo[groupby_vdo == mv].index.strftime("%y.%d.%m").tolist()[0]

def Q5(vdo_df):
    '''
        5. Compare "Sports" and "Comedy", how many days that there are more total daily views of VDO in "Sports" category than in "Comedy" category?
            - videos.csv has been loaded into memory and is ready to be utilized as vdo_df
            - The duplicate rows of vdo_df have been removed.
            - You must load the additional data from 'category_id.json' into memory before executing any operations.
            - To access 'category_id.json', use the path '/data/category_id.json'.
    '''
    id2name = dict()
    with open('/data/category_id.json', 'r') as f:
        file = json.load(f)
        for row in file['items']:
            if row['snippet']['title'] in ["Sports", "Comedy"]:
                id2name[int(row['id'])] = row['snippet']['title']
            if len(id2name) == 2:
                break
    vdo_df['types'] = vdo_df['category_id'].apply(lambda x: id2name[x] if x in id2name else "None")
    vdo_df = vdo_df.loc[vdo_df['category_id'].isin(id2name.keys()), :]
    groupby_df = vdo_df.groupby(['trending_date', 'types'])['views'].sum()
    unstacks = groupby_df.unstack("types", fill_value=0)
    return unstacks[unstacks['Sports'] > unstacks['Comedy']].shape[0]
    

