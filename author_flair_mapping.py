import sys
import pandas as pd
import numpy as np
from collections import defaultdict


AUTHORITARIAN="auth"
LIBERTARIAN="lib"
CENTRIST="centrist"
LEFT="left"
RIGHT="right"
AUTHCENTER="authcenter"
AUTHLEFT="authleft"
AUTHRIGHT="authright"
LIBCENTER="libcenter"
LIBLEFT="libleft"
LIBRIGHT="libright"


__vertical_mapping__ = { ###mapping of global categories onto the vertical dimension
    AUTHCENTER: AUTHORITARIAN,
    AUTHLEFT: AUTHORITARIAN,
    AUTHRIGHT: AUTHORITARIAN,
    CENTRIST: CENTRIST,
    LEFT: None,
    LIBCENTER: LIBERTARIAN,
    LIBLEFT: LIBERTARIAN,
    LIBRIGHT: LIBERTARIAN,
    RIGHT: None,
    LIBERTARIAN: LIBERTARIAN,
    AUTHORITARIAN: AUTHORITARIAN
}

__horizontal_mapping__ = { ###mapping of global categories onto the horizontal dimension
    AUTHCENTER: CENTRIST,
    AUTHLEFT: LEFT,
    AUTHRIGHT: RIGHT,
    CENTRIST: CENTRIST,
    LEFT: LEFT,
    LIBCENTER: CENTRIST,
    LIBLEFT: LEFT,
    LIBRIGHT: RIGHT,
    RIGHT: RIGHT,
    LIBERTARIAN: None,
    AUTHORITARIAN: None
}

__inv_horizontal_mapping__ = defaultdict(list)
for key, value in __horizontal_mapping__.items():
    __inv_horizontal_mapping__[value].append(key)
__inv_vertical_mapping__ = defaultdict(list)
for key, value in __vertical_mapping__.items():
    __inv_vertical_mapping__[value].append(key)

all_valid_flairs = [AUTHCENTER, AUTHLEFT, AUTHRIGHT, AUTHORITARIAN, CENTRIST, LEFT, LIBCENTER, LIBERTARIAN, LIBLEFT, LIBRIGHT, RIGHT]
vertical_valid_flairs = [k for k in __vertical_mapping__ if __vertical_mapping__[k] is not None]
horizontal_valid_flairs = [k for k in __horizontal_mapping__ if __horizontal_mapping__[k] is not None]

VERTICAL_DIMENSION = 'vertical'
HORIZONTAL_DIMENSION = 'horizontal'

def __map_to_valid_dimension__(input_dimension: str) -> str:
    """ Maps the input string to one of the three possible dimensions of the Political Compass: ['global', 'horizontal', 'vertical']
    """
    if input_dimension in [VERTICAL_DIMENSION, HORIZONTAL_DIMENSION, None]:
        return input_dimension
    if isinstance(input_dimension, bool):
        return None
    if input_dimension == 0 or (str(input_dimension).lower() in ['v', 'vertical', '0']):
        return VERTICAL_DIMENSION
    if input_dimension == 1 or (str(input_dimension).lower() in ['h', 'horizontal', '1']):
        return HORIZONTAL_DIMENSION
    return None


def get_mother_categories(category: str, dimension: str = None, ignore_case: bool = False) -> str|tuple[str,str]:
    """
    Given a global category, returns the corresponding category on the input "dimension". If dimension is None, returns (as a tuple) the corresponding category on both the horizontal and vertical dimensions
    """
    if ignore_case:
        category = category.lower()

    curr_dimension = __map_to_valid_dimension__(dimension)
    if curr_dimension==VERTICAL_DIMENSION:
        return __vertical_mapping__.get(category, np.nan)
    if curr_dimension==HORIZONTAL_DIMENSION:
        return __horizontal_mapping__.get(category, np.nan)

    horizontal_mother = __horizontal_mapping__.get(category, np.nan)
    vertical_mother = __vertical_mapping__.get(category, np.nan)
    return vertical_mother, horizontal_mother

def map_mother_categories(df, category_col: str, dimension: str = None, ignore_case: bool = False):
    """
    Extracts the corrisponding horizontal and vertical categories from the global category (specified on the "category_col") and adds them as columns to the input df
    """
    df = df.copy()
    curr_dimension = __map_to_valid_dimension__(dimension)
    if curr_dimension == VERTICAL_DIMENSION:
        df[VERTICAL_DIMENSION+'_'+category_column] = df[category_col].map(lambda t: get_mother_categories(t, dimension=curr_dimension, ignore_case=ignore_case))
        return df
    if curr_dimension==HORIZONTAL_DIMENSION:
        df[HORIZONTAL_DIMENSION+'_'+category_column] = df[category_col].map(lambda t: get_mother_categories(t, dimension=curr_dimension, ignore_case=ignore_case))
        return df
    
    df['vertical_'+category_col], df['horizontal_'+category_col]  = zip(*df[category_col].apply(lambda t: get_mother_categories(t, ignore_case=ignore_case)))
    return df


def map_to_default_flair(curr_flair, dimension=None, ignore_case=False):
    """
    Given an input (Political Compass) category, returns its corresponding valid category, if any, otherwise returns None
    """
    def get_all_valid_flairs(dimension):
        curr_dimension = __map_to_valid_dimension__(dimension)
        if curr_dimension == VERTICAL_DIMENSION:
            return vertical_valid_flairs
        if curr_dimension == HORIZONTAL_DIMENSION:
            return horizontal_valid_flairs
        return all_valid_flairs

    if ignore_case:
        curr_flair = curr_flair.lower()
    valid_flairs = get_all_valid_flairs(dimension)
    if curr_flair in [np.nan, None, '', ' ']:
        return np.nan
    if curr_flair in valid_flairs:
        return curr_flair
    
    flair = curr_flair.split()[-1]
    if flair not in valid_flairs:
        return np.nan
    return flair


def get_politicalcompass_author_flair(author_df, author_flair_col = 'author_flair_text'):
    """Returns the most frequent category used by the author as the default category for him
    """
    author_polcompass_df = author_df[author_df.subreddit.str.lower().isin(['politicalcompass','politicalcompassmemes', 'politicalcompassmemes2'])]
    if author_polcompass_df.empty:
        return np.nan
    ###todo    
    try:
        return author_polcompass_df[author_flair_col].agg(lambda s: s.mode().sample(1)[0])
    except:
        return np.nan
    return author_polcompass_df[author_flair_col].last(skipna=True)


def get_author_flair_by_mother_dimensions(v_flair: str, h_flair: str) -> str:
    """
    Maps the current vertical and horizontal categories to the corresponding global category, if any.
    Otherwise returns False
    """
    possible_v_flairs = __inv_vertical_mapping__.get(v_flair, None)
    possible_h_flairs = __inv_horizontal_mapping__.get(h_flair, None)
    if possible_v_flairs is None or possible_h_flairs is None:
        return False
    final_v = set(possible_v_flairs) & set(possible_h_flairs)
    if len(final_v)>1:
        raise ValueError('should never happen')
    return next(iter(final_v))


def compatible_categories(categories: list[str], dimension: str = None) -> bool:
    mother_categories = [get_mother_categories(c, dimension=dimension) for c in categories]
    unique_dimensions_df = pd.DataFrame(mother_categories).apply(lambda t: pd.Series(t.dropna().unique())).replace([np.nan], None)
    if unique_dimensions_df.empty or unique_dimensions_df.apply('count').max()>1:
        return False
    if __map_to_valid_dimension__(dimension) in [HORIZONTAL_DIMENSION, VERTICAL_DIMENSION]:
        return unique_dimensions_df.loc[0][0]
    return unique_dimensions_df.apply(lambda t: get_author_flair_by_mother_dimensions(*t), axis=1)[0]
    return unique_dimensions_df.apply(lambda t: t[0] if t.count() == 1 else '' if not t.count() else np.nan).sum()
    return pd.DataFrame(mother_categories).apply(lambda t: t.dropna().nunique()<2).all()


def get_changes(author_df, column_to_analyse, count_nan_as_change=False, handle_nans_separately=False):
    author_df = author_df.copy()
    if handle_nans_separately:
        count_nan_as_change = True
    curr_author_category_values = author_df.loc[
        (count_nan_as_change | author_df[column_to_analyse].notna()), column_to_analyse]
    if not handle_nans_separately:
        nan_value = -1  # '' if is_string_dtype(curr_author_category_values) else False if is_bool_dtype(curr_author_category_values) else -1
        curr_author_category_values = curr_author_category_values.fillna(nan_value)
    is_flair_change = (curr_author_category_values != curr_author_category_values.shift(1)).rename('is_flair_change')
    if not count_nan_as_change:
        is_flair_change.iloc[0] = False
    changes_df = pd.DataFrame(index=author_df.index)
    changes_df['is_flair_change'] = False
    changes_df.loc[curr_author_category_values.index, 'is_flair_change'] = is_flair_change
    changes_df.loc[changes_df.index[0], 'is_flair_change'] = True

    changes_df['progressive_flair_id'] = changes_df.is_flair_change.cumsum()
    return changes_df


def get_valid_authors(polcompass_df, how: int, category_col: str = 'author_flair_text', dimension: str = None, dropna: bool = True, return_cat: bool = True, ) -> pd.Series :
    UNIQUE_AUTHORS_ONLY = 0
    COMPATIBLE_CAT_AUTHORS_ONLY = 1
    ONE_DIRECTION_CHANGES_ONLY = 2
    SPLIT_BY_PERIOD = 3
    GET_ONE_LABEL_BY_AUTHOR = 4
    """
    @input how: 
        0 for authors having only 1 value for category_col
        1 for authors having only 1 compatible category (e.g. Left, AuthLeft -> mapped to AuthLeft, therefore only 1 category)
        2 for splitting authors (as multiple data points) that have multiple compatible categories but who only have 1 exact category for any given timespan (i.e. all of the category changes are one directional, the author never changes back to any of the previous categories)
        3 for splitting authors by period even if they have multiple repeated changes (e.g. Right->Center->Right->Center), by assigning to each period the most frequent label
        4 for assigning one exact label to each author, even if they have multiple different (incompatible) categories, by assigning the most frequent label
    """
    polcompass_df = polcompass_df.copy()
    if dimension is not None:
        polcompass_df[category_col] = polcompass_df[category_col].map(lambda cat: get_mother_categories(cat, dimension=dimension))
    if dropna:
        polcompass_df = polcompass_df.groupby('author').filter(
            lambda df: df[category_col].notna().any())  ##removing authors with nan category only

    if how==UNIQUE_AUTHORS_ONLY:
        #unique_values_by_author = polcompass_df.groupby('author')[category_col].nunique()
        unique_cats_by_author = polcompass_df.groupby('author')['author_flair_text_str'].agg(lambda cats: cats.dropna().unique() if dropna else cats.unique())
        valid_authors = unique_cats_by_author[unique_cats_by_author.map(len) == 1]
        return valid_authors.map(lambda v: v[0]) if return_cat else valid_authors.index
    if how==COMPATIBLE_CAT_AUTHORS_ONLY:
        unique_compatible_cats_by_author = polcompass_df.groupby('author')[category_col].apply(
            lambda values:
                unique_v if
                    len( unique_v := values.dropna().unique() if dropna else values.unique() ) == 1 or (mapped_cat := compatible_categories(unique_v, dimension=dimension)) is False
                else
                    [mapped_cat]
        )
        valid_authors = unique_compatible_cats_by_author[unique_compatible_cats_by_author.map(len) == 1]
        return valid_authors.map(lambda v: v[0]) if return_cat else valid_authors.index
    if how==ONE_DIRECTION_CHANGES_ONLY:
        compatible_authors = get_valid_authors(polcompass_df, category_col=category_col, dimension=None, how=COMPATIBLE_CAT_AUTHORS_ONLY, return_cat=False)
        changes_df = polcompass_df[~polcompass_df.index.get_level_values('author').isin(compatible_authors)]\
            .groupby('author').apply(
                lambda author_df: pd.concat([author_df, get_changes(author_df, column_to_analyse=category_col, count_nan_as_change=not dropna)], axis=1, ignore_index=False) ).droplevel(0)
        valid_changes_df = changes_df.groupby('author').filter(lambda author_df: author_df[category_col].nunique(dropna=dropna)==author_df['progressive_flair_id'].max())
        validchanges_authors_cat_by_period = valid_changes_df.groupby('author').apply(lambda author_df: author_df.groupby('progressive_flair_id').agg({'created_utc':['min','max'], category_col: 'first'})).droplevel(1)
        compatible_authors_cat_by_period = polcompass_df[polcompass_df.index.get_level_values('author').isin(compatible_authors)].groupby('author').agg(
            {'created_utc':['min','max'],
            category_col: lambda cats: unique_cat[0] if len(unique_cat := cats.dropna().unique())==1 else compatible_categories(unique_cat, dimension=dimension)}
        )
        cols_names = ['created_utc_min', 'created_utc_max', category_col]
        validchanges_authors_cat_by_period.columns, compatible_authors_cat_by_period.columns = cols_names, cols_names
        all_authors_cat_by_period = pd.concat([validchanges_authors_cat_by_period, compatible_authors_cat_by_period],axis=0)
        return all_authors_cat_by_period if return_cat else all_authors_cat_by_period.index.unique()