import json
from app import app
from flask import request, abort

@app.route("/predict", methods = ["POST"])
def predict():
    from flask import request, has_request_context
    
    if has_request_context():
        request_id = request.environ.get("HTTP_X_REQUEST_ID")
    if not request.json:
        abort(400)
    data = request.json
    texts = data.get('text')
    module_id = data.get('module_id')
    from .helpers import predict_service, get_module
    result = predict_service.delay(module_id, texts, request_id)
    return result.get()

@app.route("/train", methods = ["POST"])
def train():
    from .helpers import (select_module, train_service, get_module,
                          get_module_language, get_stopwords, get_test_ratio,
                          get_cv, get_split)
    if not request.json:
        abort(400)
    request_data = request.get_json()
    module_id = request_data.get('module_id')
    is_test = request_data.get('is_test')
    data = select_module(module_id)
    module = get_module(module_id)
    language = get_module_language(language_id=1) # temproary variable right now because mnew models doesn't want param.
    # stopwords = get_stopwords(module_id)
    # test_ratio = get_test_ratio(module_id)
    # cv = get_cv(module_id)
    # split = get_split(module_id)
    test_ratio = 0.2
    cv = 3
    split =  100
    
    used_stopwords=module.stopword
    used_stemming=module.stemming
    used_remove_numbers=module.remove_numbers
    used_deasciify=module.deasciify
    used_remove_punkt=module.remove_punkt
    used_lowercase=module.lowercase
    
    initial_params = {
    "stopwords":module.stopword,
    "stemming":module.stemming,
    "remove_numbers":module.remove_numbers,
    "deasciify":module.deasciify,
    "remove_punkt":module.remove_punkt,
    "lowercase":module.lowercase,
    "TEST_RATIO" : test_ratio,
    "CV" : cv,
    "SPLIT" : split
    }
    texts = [i.text for i in data]
    labels = [i.label for i in data]
    train_service.delay(module_id, texts, labels, language, split, initial_params, is_test = is_test)
    response = app.response_class(
            response = json.dumps({"data": True}),
            status = 200,
            mimetype = 'application/json'
    )
    return response
