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
    # TODO: Paste your code here
    vdo_df = pd.read_csv("/data/videos.csv")
    vdo_df.drop_duplicates(inplace=True)
    sol = vdo_df.shape[0]
    return sol

def Q2(vdo_df):
    '''
        2. How many VDO that have "dislikes" more than "likes"? Make sure that you count only unique title!
            - videos.csv has been loaded into memory and is ready to be utilized as vdo_df
            - The duplicate rows of vdo_df have been removed.
    '''
    lmd = vdo_df[vdo_df["dislikes"] > vdo_df["likes"]]
    return lmd["title"].nunique()

def Q3(vdo_df):
    '''
        3. How many VDO that are trending on 22 Jan 2018 with comments more than 10,000 comments?
            - videos.csv has been loaded into memory and is ready to be utilized as vdo_df
            - The duplicate rows of vdo_df have been removed.
            - The trending date of vdo_df is represented as 'YY.DD.MM'. For example, January 22, 2018, is represented as '18.22.01'.
    '''
    dt = "18.22.01"
    vdo_comments = vdo_df[(vdo_df["trending_date"] == dt) & (vdo_df["comment_count"] > 10000)]
    return vdo_comments['video_id'].count()

def Q4(vdo_df):
    '''
        4. Which trending date that has the minimum average number of comments per VDO?
            - videos.csv has been loaded into memory and is ready to be utilized as vdo_df
            - The duplicate rows of vdo_df have been removed.
    '''
    avg_comment = vdo_df.groupby("trending_date")["comment_count"].mean()
    date = avg_comment.idxmin()
    return  date

def Q5(vdo_df):
    '''
        5. Compare "Sports" and "Comedy", how many days that there are more total daily views of VDO in "Sports" category than in "Comedy" category?
            - videos.csv has been loaded into memory and is ready to be utilized as vdo_df
            - The duplicate rows of vdo_df have been removed.
            - You must load the additional data from 'category_id.json' into memory before executing any operations.
            - To access 'category_id.json', use the path '/data/category_id.json'.
    '''
    with open('/data/category_id.json') as file:
        cate = json.load(file)
        cate_list = []
        for item in cate['items']:
            cate_list.append({"category_id":int(item['id']),"cate_name":str(item['snippet']['title'])})
        df_cate = pd.DataFrame(cate_list)
        merge_df = vdo_df.merge(df_cate, left_on="category_id", right_on="category_id", how="inner", sort=True)
        
        groupby_cate = merge_df.groupby(["trending_date", "cate_name"])["views"].sum()
        df_extract = groupby_cate.unstack(fill_value=0)
        ans = (df_extract["Sports"] > df_extract["Comedy"]).sum()
        return ans
        # ans = (df_extract["Sports"] > df_extract["Comedy"]).sum()
        # return ans 
        # sum_per_cate = groupby_cate['views'].sum()
        # for name, col in sum_per_cate.values
    # return None