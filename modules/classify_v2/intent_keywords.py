"""
@author: funda
@author: uğuray

"""

from param_config import PREP_PARAMS
from preprocess import Preprocessor

import requests 
import numpy as np
import pandas as pd

from collections import Counter 
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def nmf(data):
    
    # preprocess
    labels = data['label']
    
    # get instances for each category
    cats = list(set(labels))
    cats_instances = dict.fromkeys(cats, [])
    for text, label in zip(data['after'].tolist(), labels):
        text_ = text.split(" ")
        txt = []
        for word in text_:
            txt.append(word)
        cats_instances[label] = cats_instances[label] + [" ".join(txt)]
        
    nmf_keywords = dict.fromkeys(cats, [])
    for category, txts in cats_instances.items():

        # nmf model
        nmf_model = NMF(n_components=1, random_state=1, alpha=.1, l1_ratio=.5)

        # tfidf vectorizer
        vectorizer = TfidfVectorizer(preprocessor=None, max_features=200)
        tf_idf = vectorizer.fit_transform(txts)
        
        feature_names = vectorizer.get_feature_names()
        nmf_model = nmf_model.fit(tf_idf)
        nmf_weights = nmf_model.components_

        # top 20 words for each category
        n_top_words = 20
        topics_keywords = {}
        for topic_idx, topic in enumerate(nmf_weights):
            topical_terms = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
            topics_keywords[topic_idx] = topical_terms
            nmf_keywords[category] = topics_keywords
            
    return nmf_keywords


def lda(data):
    
    # preprocess
    labels = data['label']
    
    # get instances for each category
    cats = list(set(labels))
    cats_instances = dict.fromkeys(cats, [])
    for text, label in zip(data['after'].tolist(), labels):
        cats_instances[label] = cats_instances[label] + [text]
        
    lda_keywords = dict.fromkeys(cats, [])
    for category, txts in cats_instances.items():

        # lda model
        lda_model = LatentDirichletAllocation(n_components=1)

        # tfidf vectorizer
        vectorizer = TfidfVectorizer(preprocessor=None, max_features=200)
        tf_idf = vectorizer.fit_transform(txts)

        feature_names = vectorizer.get_feature_names()
        lda_model = lda_model.fit(tf_idf)
        lda_weights = lda_model.components_
        
        # top 20 words for each category
        n_top_words = 20
        topics_keywords = {}
        for topic_idx, topic in enumerate(lda_weights):
            topical_terms = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
            topics_keywords[topic_idx] = topical_terms
            lda_keywords[category] = topics_keywords
    
    return lda_keywords


def find_most_freq_words(data, classes):
    
    # -----------------------
    # collections - Counter
    # most frequent words 
    # -----------------------
    
    # most_freq_words --> {key:class, value:list of tuple(word, frequency)}
    most_freq_words = {}
    for cls_ in classes:
        
        data_ = data[data['label'] == cls_] 
        text_lst = [text.split(" ") for text in data_['after'].tolist()]
        texts = [word for lst in text_lst for word in lst]
        
        words = []
        for word in texts:
            words.append(word)
        
        Counter_ = Counter(words)
        most_occur = Counter_.most_common(20) 
        most_freq_words[cls_] = most_occur
        
    return most_freq_words
    

def find_intersection(classes, from_count, from_nmf, from_lda):
    
    intersect_words = {}
    for clss in classes:
        # get words from each method
        words_from_counter = [word_tuple[0] for word_tuple in from_count[clss]]
        words_from_nmf = from_nmf[clss][0]
        words_from_lda = from_lda[clss][0]
        
        intersect_words[clss] = list(set(words_from_counter) & set(words_from_nmf) & set(words_from_lda))
    
    return intersect_words


def change_words(data, clss, fraction, words_to_change, new_word):
    '''
    clss = 'futbol'
    fraction = 0.5
    words_to_change = intersect_words['futbol']
    new_word = "basketbol"
    '''

    data_ = data[data['label']==clss]
    
    # find samples contain words_to_change
    for index, row in data_.iterrows():
        text = row['after'].split(" ")
        if set(text) & set(words_to_change):
            continue
        else:
            data_.drop(index, inplace=True)
                
    # random samples from data
    data_ = data_.sample(frac = fraction)      

    # change words_to_change to new_word
    texts = [text.split(" ") for text in data_['after'].tolist()]
    texts_new = []
    for idx1, text in enumerate(texts):
        for idx2, word in enumerate(texts[idx1]):
            if texts[idx1][idx2] in words_to_change:
                texts[idx1][idx2] = new_word 
            else:
                texts[idx1][idx2] = word
    texts_new.append(texts)
    
    texts_new = [wrd for lst in texts_new for wrd in lst]
    texts_changed = [" ".join(text) for text in texts_new]

    data_['text_changed'] = texts_changed
     
    return data_


if __name__ == "__main__":

    # read data
    xlsx_data = pd.read_excel(DATA_FOLDER+DATA_NAME, usecols=['text','label'])
    xlsx_data.dropna(axis=0, inplace=True)
    
    # preprocess
    data = Preprocessor(xlsx_data, PREP_PARAMS)
    data['label'] = xlsx_data['label']
    classes = data['label'].unique().tolist()
    
    # most frequent words with Counter
    # most_freq_words --> {key:class, value:list of tuple(word, frequency)}
    most_freq = find_most_freq_words(data, classes)
    
    # keywords with NMF
    # keywords_nmf --> {key:classes, value: dict{key:0, value: list of words}} 
    keywords_nmf = nmf(data)
    
    # keywords with LDA
    keywords_lda = lda(data)
    
    # find intersection of words from {Counter, NMF, LDA}
    intersect_words = find_intersection(classes, most_freq, keywords_nmf, keywords_lda)
    
    # change intersect to new_word
    # get new_df
    new = change_words(data, "bordür ve tretuvar", 1, intersect_words['bordür ve tretuvar'], "bukelimeyikaldırdık")
       
    # add new_df to data
    slice1 = data[['after','label']].rename(columns={'after': 'text'})
    slice2 = new[['text_changed','label']].rename(columns={'text_changed': 'text'})

    final_data = pd.concat([slice1, slice2]).reset_index(drop=True)
        
    # save final_df     
    final_data.to_excel('/home/uguray/classify/classify_zero/data/____.xlsx')
