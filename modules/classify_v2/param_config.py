"""
@author: funda
@author: uğuray

"""
# parameters for preprocess
PREP_PARAMS = {
    
    'lowercase': True,
    'remove_urls': True,
    'remove_punkts': True,
    'remove_numbers': True,
    'remove_stopwords': True, # add
    'slang_look_up': True,
    'contraction_look_up': True,
    'remove_after_apostrophe': True,
    'remove_single_char_word':True,
    'remove_repeated_chars': True,
    'zemberek_stemmer': False, # add
    'zemberek_spellcheck': False, # add
    'spellcheck_selected_word_types': ['Unk']
    
}

unwanted_chars = "«‹»›„‟❟❝❞❮❯⹂〝〞〟＂©™®·¯·●♣🌿“”█√"


# parameters for TfidfVectorizer
VECT_PARAMS = {
    
    'word_ngram_range' : (1,2), 
    'max_features' :  10000
    
}


# data
DATA_NAME = 'cog_module_1k.xlsx'
#DATA_NAME = 'cog_module_100k.xlsx'


# train_test split
SPLIT_RATIO_CV = {
	"SPLIT" : 100,
	"TEST_RATIO" : 0.2,
	"CV" : 3
}

# folders
DATA_FOLDER = "data/"
MODEL_FOLDER = "models/"
VECTORIZER_FOLDER = "vectorizers/"
CLANA_FOLDER = "clana/"
model_name = "model.b"
vectorizer_name = "vectorizer.b"
# model and vectorizer save
MODEL_SAVE = False
