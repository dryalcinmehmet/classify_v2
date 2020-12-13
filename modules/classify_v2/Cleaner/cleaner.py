"""
Created on Wed Apr 22 16:13:07 2020

@author: funda
@author: uğuray

"""

import os
import re
import string
import requests
from urllib import parse

# https://docs.python.org/3/library/urllib.parse.html
from urllib.parse import urlparse

from unicode_tr import unicode_tr
from . import tr_stopword_list as stopwords
from .slangs_tr import slangs_dict
from .contractions_tr import contractions_dict

from .param_config import PREP_PARAMS, unwanted_chars


selected_word_types = PREP_PARAMS['spellcheck_selected_word_types']


ZEMBEREK_URL = os.getenv('ZEMBEREK_URL', 'http://localhost:4567')


def tokenize(sentence):
    tokens = sentence.split(' ')
    return tokens


def lowercase(tokens):
    tokens = [unicode_tr(token).lower() for token in tokens]
    return tokens


def remove_urls(tokens):
    tokens = [token for token in tokens if not urlparse(token).scheme if len(token) > 0]
    return tokens


def remove_stopwords(tokens):
    tokens = [token for token in tokens if token not in stopwords.tr_stopwords()]
    tokens = [token for token in tokens if len(token) > 0]
    return tokens


def slang_look_up(tokens):
    new_text = []
    for token in tokens:
        if token in slangs_dict:
            new_text.append(slangs_dict[token])
        else:
            new_text.append(token)
    return new_text


def contraction_look_up(tokens):
    new_text = []
    for token in tokens:
        if token in contractions_dict:
            new_text.append(contractions_dict[token])
        else:
            new_text.append(token)
    return new_text


def remove_after_apostrophe(tokens):
    # for chars --> ',’,‘,´,‛,❛,❜
    regex_pattern = re.compile(r"(['|’|‘|´|‛|❛|❜][\w]+)")
    tokens = [re.sub(regex_pattern, '', token) for token in tokens \
              if len(re.sub(regex_pattern, '', token)) > 0]
    return tokens


def remove_numbers(tokens):
    regex_pattern = re.compile(r'[0-9]')
    tokens = [re.sub(regex_pattern, '', token) for token in tokens \
              if len(re.sub(regex_pattern, '', token)) > 0]
    return tokens


def remove_punkts(tokens):
    chars = string.punctuation + unwanted_chars
    regex_pattern = re.compile(r'['+chars+']')
    tokens = [re.sub(regex_pattern, ' ', token.replace("\\", "")).strip() \
              for token in tokens if len(re.sub(regex_pattern, '', token)) > 0]
    return tokens


def remove_single_char_word(tokens):
    tokens = [token for token in tokens if len(token) > 1]
    return tokens


def remove_repeated_chars(tokens):
    regex_pattern = re.compile(r'(.)\1+')
    tokens = [regex_pattern.sub(r'\1\1', token) for token in tokens \
              if len(regex_pattern.sub(r'\1\1', token)) > 0]
    return tokens


def zemberek_stemmer(tokens):
    stemmed_tokens = []
    for token in tokens:
        headers = {'Content-Type': 'application/x-www-form-urlencoded; charset=utf-8'}
        data = {"word": token}
        response = requests.post(parse.urljoin(ZEMBEREK_URL, 'stems'), headers=headers, data=data)
        
        try:
            response_json = response.json()
            stem = response_json['results'][0]['stems'][0]
            stemmed_tokens.append(stem)
        except:
            stemmed_tokens.append(token)
            continue
    
    return stemmed_tokens    


def zemberek_spellcheck(tokens):
    # zemberek pos-tagging
    pos_tags = {}
    for token in tokens:
        headers = {'Content-Type': 'application/x-www-form-urlencoded; charset=utf-8'}
        data= {'sentence': token}
        response = requests.post(parse.urljoin(ZEMBEREK_URL, "find_pos"), \
                                 headers=headers, data=data)
        
        # get pos tags
        if response.json() is not None:
            pos = response.json()[0]['pos']
            pos_tags[token] = pos             

    # apply spellcheck for selected word types
    tokens_ = []                  
    for token in tokens:        
        # if pos_tags[token] in selected_word_types:
        if pos_tags[token] in selected_word_types:
            # is spelling is correct ?
            headers = {'Content-Type': 'application/x-www-form-urlencoded; charset=utf-8'}
            data= {'word': token}
            response = requests.post(parse.urljoin(ZEMBEREK_URL, 'spelling_check'), \
                                      headers=headers, data=data)
            response_json = response.json()
            
            # if spelling is correct
            spell = response_json['is_correct']
            if spell:
                tokens_.append(token)
            
            # spelling is incorrect --> get suggestions
            else:
                response2 = requests.post(parse.urljoin(ZEMBEREK_URL, 'spelling_suggestions'), \
                                           headers=headers, data=data)
                response_json2 = response2.json()
                
                if len(response_json2['suggestions']) > 0:
                    suggest = response_json2['suggestions'][0]
                    tokens_.append(suggest)
                else:
                    tokens_.append(token)
        
        else:             
            tokens_.append(token)            

    return tokens_   
