"""
@author: funda
@author: uğuray

"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from .Cleaner.cleaner import *

from .param_config import PREP_PARAMS


def Preprocessor(data, params):
    '''
    Call the text processing functions if True.

    Parameters
    ----------
    data : DataFrame[Text, Label] or str
    params : dict

    Returns
    -------
    cleaned : DataFrame[Before, After, Label] or str

    '''

    # preproces steps
    functions = [key for (key, value) in params.items() if value==True]
    
    # if data type == str
    if isinstance(data, str):        
        # tokenization
        tokens = tokenize(data)
        
        # apply selected steps
        cleaned = []
        for func in tqdm(functions):
            tokens = globals()[func](tokens)      
        cleaned = ' '.join(tokens)
    
    # data type == DataFrame
    else:
        # tokenization
        tokens = pd.DataFrame(data['text'].map(lambda x: tokenize(x)))
        
        # apply selected steps
        cleaned = pd.DataFrame({'before': data['text'], \
                                'after': tokens['text'], \
                                'label': data['label']}
                )
        
        for func in tqdm(functions):
            cleaned['after'] = cleaned['after'].map(lambda x: globals()[func](x))      
        cleaned['after'] = cleaned['after'].str.join(' ')   
        cleaned = cleaned[['before','after','label']]
        
        # delete row if 'after' column is empty
        cleaned['after'].replace("", np.nan, inplace=True)
        cleaned.dropna(subset=['after'], inplace=True)
        
    return cleaned

