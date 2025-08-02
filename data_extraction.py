#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import random, os, glob, json
import default_config
from datetime import datetime,timezone,timedelta
from utils import is_daylight_saving

def get_whole_filelist(path=default_config.json_path, file_format='', filename_string='') -> list[str]:
    file_format = file_format.strip()
    if file_format:
        if not file_format.startswith('.'):
            file_format = '.'+file_format
    searching_pattern = path+'*{}*{}'.format(filename_string, file_format)
    file_list = glob.glob(searching_pattern)
    return file_list
            
def filter_out_bots(filenames: list[str], path: str = None, case: bool = True) -> pd.Series:
    #path = path[:-1]+'\\'
    
    bot_pattern = r"((([-_\W]|^)b)|B)[oO0][tT]([^a-z]|$)"
        
    bot_pattern_end = r"[bB][oO0][tT]$"
    #splitted_file_list = list(map(lambda filename: (filename[0],filename[1].split(filename[0])[1],filename[2]), [ [os.path.dirname(x)]+list(os.path.splitext(x)) for x in filenames]))    
    #dirs, names, formats = list(zip(*splitted_file_list))

    df = pd.DataFrame({'whole_filename':filenames})
    df['author_name'] = df['whole_filename'].map(lambda x: os.path.splitext(x)[0].split(os.path.dirname(x))[1])
    
    #df = pd.DataFrame({'whole_filename':filenames, 'author_name':names})
    return df.loc[~(df['author_name'].str.contains(pat=bot_pattern, regex =  True, case=case)) & (~df['author_name'].str.contains(pat=bot_pattern_end, regex=True, case=False)), 'whole_filename']
    
    
def default_dict_none_values():
    return {None:-1, np.nan:-2}


def standard_none_mapping() -> dict[str, str]:
    """
    Maps the nan value to "none_value:nan:none_value" in order to avoid casual random matching of strings with "nan"
    """
    standard_none_dict = default_dict_none_values()
    default_json_none_mapping_dict = {none_value: (':none_value:'+str(none_value)+':none_value:', standard_none_dict[none_value]) for none_value in standard_none_dict}
    return default_json_none_mapping_dict


def store_global_id_dict(id_dict: dict, filename: str, directory: str = './data/dataset_whole/', custom_none_mapping: dict[str, str] = None) -> None:
    if custom_none_mapping:
        id_dict = id_dict.copy()
        for k in custom_none_mapping.keys():
            id_dict[custom_none_mapping[k]]=id_dict.pop(k)
    # create json object from dictionary
    js = json.dumps(id_dict)
    # open file for writing, "w" 
    f = open(directory+filename,"w")
    # write json object to file
    f.write(js)
    # close file
    f.close()
    
    


def load_global_id_dict(filename: str, directory: str = './data/dataset_whole/', custom_none_mapping: dict[str, str] = None) -> dict:
    with open(directory+filename) as f_in:
        loaded_dict = json.load(f_in)
    if custom_none_mapping:
        loaded_dict = loaded_dict.copy()
        for k in custom_none_mapping.keys():
            loaded_dict[custom_none_mapping[k]]=loaded_dict.pop(k)
        
    return loaded_dict
    

def generate_dict_categorical_column(categorical_series: pd.Series, existing_dict: dict, copy = True) -> dict:
    if not isinstance(categorical_series, pd.Series):
        if isinstance(categorical_series, (list, np.ndarray)):
            categorical_series = pd.Series(categorical_series)
        else:
            raise ValueError('categorical_series must be either a pandas Series or a list/array type')
            return
    new_values = [curr_id for curr_id in categorical_series.unique() if curr_id not in existing_dict.keys()]
    
    if copy:
        existing_dict = existing_dict.copy()
    
    #standard_dict = standard_id_dict_none_values()
    #existing_dict |= standard_dict
    if len(new_values):
        max_value = len(existing_dict)
        curr_len = len(new_values)
        curr_values_dict = dict(zip(new_values, range(max_value, max_value+curr_len)))
        #inv_dict = {curr_id_dict[k]:k for k in curr_id_dict.keys()}
        existing_dict |= curr_values_dict
    return existing_dict


def generate_subreddit_dict(df, existing_dict, subreddit_id_col = 'subreddit_id', subreddit_col = 'subreddit', copy=True):
    new_values = df.loc[~df[subreddit_id_col].isin(existing_dict.keys()), [subreddit_id_col, subreddit_col]]
    if new_values.empty:
        return existing_dict
    if copy:
        existing_dict = existing_dict.copy()
    new_values = new_values.groupby(subreddit_id_col)[subreddit_col].first().to_dict()
    existing_dict |= new_values
    return existing_dict
    



def map_dtypes(df: pd.DataFrame, columns: list[str], dicts: list[dict] = [], load_disk: bool = False, map_categoricals: bool = False) -> pd.DataFrame:
    
    columns = pd.Series(range(len(columns)), columns)
    df = df.copy()
    if 'num_comments' in columns:
        df['num_comments'] = df['num_comments'].astype('int')
        columns.drop('num_comments', inplace=True)
    if 'over_18' in columns:
        df['over_18'] = df['over_18'].astype('bool')
        columns.drop('over_18', inplace=True)
    if 'is_self' in columns:
        df['is_self'] = df['is_self'].astype('bool')
        columns.drop('is_self', inplace=True)
    if 'stickied' in columns:
        df['stickied'] = df['stickied'].astype('bool')
        columns.drop('stickied', inplace=True)
    if 'score' in columns:
        df['score'] = df['score'].astype('int')           
        columns.drop('score', inplace=True)
    
    
    if not len(columns):
        return df
                        
    if 'post_id' in columns:
        if load_disk:
            whole_post_id_dict = load_global_id_dict(default_config.post_id_filename)
        else:
            whole_post_id_dict = dicts[columns.index.get_loc('post_id')]
        #whole_post_id_dict = generate_dict_categorical_column(categorical_series=df['post_id'], existing_dict=whole_post_id_dict)
        
        df.loc[:,'post_id'] = df['post_id'].map(whole_post_id_dict)#.astype('int')
        df['post_id'] = df['post_id'].astype('int', errors='ignore')
        
        
    if 'comment_id' in columns:
        if load_disk:
            whole_comment_id_dict = load_global_id_dict(default_config.comment_id_filename)
        else:
            whole_comment_id_dict = dicts[columns.index.get_loc('comment_id')]
        
        df.loc[:,'comment_id'] = df['comment_id'].map(whole_comment_id_dict)
        df['comment_id'] = df['comment_id'].astype('int', errors='ignore')
  
    
    if 'subreddit_id' in columns:
        if load_disk:
            whole_subreddit_int_dict = load_global_id_dict(default_config.subreddit_id_filename)
        else:
            whole_subreddit_int_dict = dicts[columns.index.get_loc('subreddit_id')]
        
        df.loc[:,'subreddit_id'] = df['subreddit_id'].map(whole_subreddit_int_dict)
        df['subreddit_id'] = df['subreddit_id'].astype('int', errors='ignore')

        if 'subreddit' in columns:
            #try:
            if load_disk:
                whole_subreddit_to_subredditid_filename = load_global_id_dict(default_config.subreddit_to_subredditid_filename)
            else:
                whole_subreddit_to_subredditid_filename = dicts[columns.index.get_loc('subreddit')]
            
            inv_subreddit_to_subredditid_filename = {whole_subreddit_to_subredditid_filename[k]:k for k in whole_subreddit_to_subredditid_filename.keys()}
            df.loc[:,'subreddit'] = df['subreddit'].map(inv_subreddit_to_subredditid_filename)


    if 'author_flair_text' in columns:
        if load_disk:
            whole_author_flair_dict = load_global_id_dict(default_config.author_flair_id_filename)
        else:
            whole_author_flair_dict = dicts[columns.index.get_loc('author_flair_text')]
        
        df.loc[:,'author_flair_text'] = df['author_flair_text'].map(whole_author_flair_dict)
        if map_categoricals:
            df['author_flair_text'] = df['author_flair_text'].astype('category', errors='ignore')
        else:
            df['author_flair_text'] = df['author_flair_text'].astype('int', errors='ignore')
        
        
    if 'author' in columns:
        if load_disk:
            whole_author_dict = load_global_id_dict(default_config.author_id_filename)
        else:
            whole_author_dict = dicts[columns.index.get_loc('author')]
        
        df.loc[:,'author'] = df['author'].map(whole_author_dict)
        if map_categoricals:
            df['author'] = df['author'].astype('category', errors='ignore')
        else:
            df['author'] = df['author'].astype('int', errors='ignore')    
        
        
    if 'link_id' in columns:
        if load_disk:
            whole_link_id_dict = load_global_id_dict(default_config.link_id_filename)
        else:
            whole_link_id_dict = dicts[columns.index.get_loc('link_id')]
        
        df.loc[:,'link_id'] = df['link_id'].map(whole_link_id_dict)
        df['link_id'] = df['link_id'].astype('int', errors='ignore') 
        
        
    if 'parent_id' in columns:
        if load_disk:
            whole_parent_id_dict = load_global_id_dict(default_config.parent_id_filename)
        else:
            whole_parent_id_dict = dicts[columns.index.get_loc('parent_id')]
        
        df.loc[:,'parent_id'] = df['parent_id'].map(whole_parent_id_dict)
        df['parent_id'] = df['parent_id'].astype('int', errors='ignore')
        
        
        
    if 'created_utc' in columns:
        df['created_utc'] = df['created_utc'].map(lambda x: datetime.fromtimestamp(x)).map(lambda x: \
                                                                                           x-timedelta(hours=1+int(is_daylight_saving(x))))
        
        
    if 'date' in columns:
        df['date'] = df['date'].map(lambda x: pd.to_datetime(x, dayfirst=True))

        
        
    return df



def get_expanded_df(df, main_col, remove_all_missing=True, map_dtypes=True, columns=False):
    expanded_df = df[main_col].sort_index().explode().apply(pd.Series)
    if columns:
        if not isinstance(columns, (list, np.ndarray, pd.Series)):
            columns = [columns]
        try:
            expanded_df = expanded_df.loc[:, columns]
        except KeyError:
            curr_cols = [c for c in columns if c in expanded_df.columns]
            if not curr_cols:
                return pd.DataFrame()
            expanded_df = expanded_df.loc[:, curr_cols]
    
    
    if remove_all_missing:
        expanded_df = expanded_df.loc[:, expanded_df.notna().any(axis=0)]
        expanded_df = expanded_df.loc[expanded_df.notna().any(axis=1)]
        
    if map_dtypes:
        if not columns:
            if main_col == 'posts':
                columns = default_config.default_posts_columns
            elif main_col == 'comments':
                columns = default_config.default_comments_columns
            else:
                raise ValueError('Columns must be declared')
        expanded_df = map_dtypes(expanded_df, columns = columns)        
    
    return expanded_df.rename({'id':main_col[:-1]+'_id'},axis=1)

    




def load_dataframes(n_authors, load_posts=True, load_comments=True, path='./data/dataset_whole/raw_data/', remove_bots=True, verbose=True):

    whole_posts_df, whole_comments_df = pd.DataFrame(), pd.DataFrame()
    file_list, wrong_files  = get_whole_filelist(path),[]
    
    if not isinstance(n_authors, int):
        if not isinstance(n_authors, (list, np.ndarray)):
            n_authors = [n_authors]
        file_list = list(map(lambda x: path+x if x.endswith('.json') else path+x+'.json', n_authors))
        
    if remove_bots:
        file_list = filter_out_bots(file_list, path).to_list()
    if isinstance(n_authors, int):
        #json_pattern = os.path.join(path,'*.json')
        if n_authors>len(file_list):
            print('Warning: the number of desired authors is higher than the number of available ones. Will return all the available authors')
        else:    
            file_list = random.sample(file_list, k=n_authors)
        
    
    i = 0
    for file in file_list:
        try:
            if verbose:
                print("Loading %d-th file: " %i, file)
            curr_df = pd.read_json(file)
            if load_posts:
                curr_posts_df = get_expanded_df(curr_df, main_col='posts', columns=False, map_dtypes=False, remove_all_missing=True)
                whole_posts_df = pd.concat([whole_posts_df, curr_posts_df], ignore_index = True)
                del(curr_posts_df)
            if load_comments:
                curr_comments_df = get_expanded_df(curr_df, main_col='comments', columns=False, map_dtypes=False, remove_all_missing=True)
                whole_comments_df = pd.concat([whole_comments_df, curr_comments_df], ignore_index = True)
                del(curr_comments_df)
            del(curr_df)

            #print(posts_df.memory_usage())
            i+=1
            
        except ValueError:
            print('Error loading: ', file)
            wrong_files.append(file)
        
    return whole_posts_df, whole_comments_df
   
    
    
def get_posts_df_old(df, post_col='posts', remove_all_missing=True, map_dtypes=True, columns=False):
    posts_df = df[post_col].sort_index().explode().apply(pd.Series)
    
    if columns:
        if not isinstance(columns, (list, np.ndarray, pd.Series)):
            columns = [columns]
        try:
            posts_df = posts_df.loc[:, columns]
        except KeyError:
            curr_cols = [c for c in columns if c in posts_df.columns]
            if not curr_cols:
                return pd.DataFrame()
            posts_df = posts_df.loc[:, curr_cols]
    
    
    if remove_all_missing:
        posts_df = posts_df.loc[:, posts_df.notna().any(axis=0)]
        posts_df = posts_df.loc[posts_df.notna().any(axis=1)]
        
    if map_dtypes:
        if not columns:
            columns = default_config.default_posts_columns
        posts_df = map_dtypes(posts_df, columns = columns)        
    
    return posts_df



def get_comments_df_old(df, comments_col='comments', remove_all_missing=True, map_dtypes=True, columns=False):
    comments_df = df[comments_col].sort_index().explode().apply(pd.Series)
    
    if columns:
        if not isinstance(columns, (list, np.ndarray, pd.Series)):
            columns = [columns]
        try:
            comments_df = comments_df.loc[:, columns]
        except KeyError:
            curr_cols = [c for c in columns if c in comments_df.columns]
            if not curr_cols:
                return pd.DataFrame()
            comments_df = comments_df.loc[:, curr_cols]
    
    if remove_all_missing:
        comments_df = comments_df.loc[:, comments_df.notna().any(axis=0)]
        comments_df = comments_df.loc[comments_df.notna().any(axis=1)]
        
    if map_dtypes:
        if not columns:
            columns = default_config.default_comments_columns
        comments_df = map_dtypes(comments_df, columns = default_config.default_comments_columns)        
    
    return comments_df
    
def load_dataframes_old(n_authors, load_posts=True, load_comments=True, path='./data/dataset_whole/raw_data/'):

    whole_posts_df, whole_comments_df = pd.DataFrame(), pd.DataFrame()
    file_list, wrong_files  = [],[]
    
    if not isinstance(n_authors, int):
        if not isinstance(n_authors, (list, np.ndarray)):
            n_authors = [n_authors]
        
        file_list = list(map(lambda x: path+x if x.endswith('.json') else path+x+'.json', n_authors))
            
    else:
        json_pattern = os.path.join(path,'*.json')
        file_list = random.sample(glob.glob(json_pattern), k=n_authors)
        
    
    i = 0
    for file in file_list:
        try:
            print("Loading %d-th file: " %i, file)
            curr_df = pd.read_json(file)
            if load_posts:
                curr_posts_df = get_posts_df_old(curr_df, columns=False, map_dtypes=False, remove_all_missing=True)
                whole_posts_df = pd.concat([whole_posts_df, curr_posts_df], ignore_index = True)
                del(curr_posts_df)
            if load_comments:
                curr_comments_df = get_comments_df_old(curr_df, columns=False, map_dtypes=False, remove_all_missing=True)
                whole_comments_df = pd.concat([whole_comments_df, curr_comments_df], ignore_index = True)
                del(curr_comments_df)
            del(curr_df)

            #print(posts_df.memory_usage())
            i+=1
            
        except ValueError:
            print('no, incorrect: ', file)
            wrong_files.append(file)
        
    return whole_posts_df, whole_comments_df


                        
def generate_dict_categorical_column_old(categorical_series, existing_dict={}, copy = True):
    if not isinstance(categorical_series, pd.Series):
        if isinstance(categorical_series, (list, np.ndarray)):
            categorical_series = pd.Series(categorical_series)
        else:
            raise ValueError('categorical_series must be either a pandas Series or a list/array type')
            return
    new_values = [curr_id for curr_id in categorical_series.unique() if curr_id not in existing_dict.keys()]
    
    if not existing_dict:
        existing_dict = {'max':0, np.nan:-1, None:-2}
    else:
        if not 'max' in existing_dict.keys():
            raise ValueError('existing_dict must have "max" in its keys')
            return
    if copy:
        existing_dict = existing_dict.copy()
    if len(new_values):
        max_value = existing_dict['max']
        curr_len = len(new_values)
        curr_values_dict = dict(zip(new_values, range(max_value, max_value+curr_len)))
        #inv_dict = {curr_id_dict[k]:k for k in curr_id_dict.keys()}
        existing_dict |= curr_values_dict
        existing_dict['max'] += curr_len
    return existing_dict