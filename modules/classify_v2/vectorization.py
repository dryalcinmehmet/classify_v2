"""
@author: funda
@author: uğuray

"""

import os
import glob
import pickle 
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from .param_config import VECT_PARAMS,PREP_PARAMS, VECTORIZER_FOLDER


def train_vectorizer(data = False, train = False, test = False):
    '''
    Convert a collection of raw documents to a matrix of TF-IDF features.

    Parameters
    ----------
    train : DataFrame[Before, After, Label]
    test : DataFrame[Before, After, Label]

    Returns
    -------
    train_tfidf : np.float object
    test_tfidf : np.float object

    '''
    
    # parameters for TfidfVectorizer
    word_ngrams = VECT_PARAMS['word_ngram_range']
    feature_num = VECT_PARAMS['max_features']
    
    # vectorizer
    vectorizer = TfidfVectorizer(ngram_range = word_ngrams, max_features = feature_num)
    
    # get features
    # features = vectorizer.get_feature_names()
    
    if len(data) != 0:
        data_tfidf = vectorizer.fit_transform(data['after'].tolist())
        return data_tfidf, vectorizer
    
    else:
        train_tfidf = vectorizer.fit_transform(train['after'].tolist())
        test_tfidf = vectorizer.transform(test['after'].tolist())
        # print(train_tfidf.shape, test_tfidf.shape)
        return train_tfidf, test_tfidf, vectorizer


def predict_vectorizer(data, vectorizer_file):
    '''
    Transform a collection of test documents to a matrix with TF-IDF features.

    Parameters
    ----------
    data : DataFrame[Before, After] 
    vectorizer_file : file

    Returns
    -------
    test_tfidf : np.float object

    '''
 
    #get vectorizer from vectorizers folder
    if  vectorizer_file:
        vectorizer = vectorizer_file

    else:
        VECTORIZER_FOLDER = "models/"
        list_of_files = glob.glob(VECTORIZER_FOLDER+'vectorizer_*')
        latest_file = max(list_of_files, key=os.path.getctime)
        # print("Loading vectorizer : {}".format(latest_file))
        vectorizer = pickle.load(open(latest_file, 'rb'))


    # if type data == str
    if isinstance(data, str):
        test_tfidf = vectorizer.transform([data])
    # data == DataFrame
    else:
        test_tfidf = vectorizer.transform(data['after'].tolist())

    # test tfidf matrix dimension
    # print(test_tfidf.shape)
    
    return test_tfidf

