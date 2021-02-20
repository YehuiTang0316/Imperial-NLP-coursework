import re

def get_edited_df(df):
    df_copy = df.copy(deep=True)
    df_copy['edited'] = df_copy['original']
    df_copy['edited'] = df_copy.apply(lambda x: re.sub(r"(<[^>]+>)", x['edit'], x['edited']), axis=1)
    return df_copy