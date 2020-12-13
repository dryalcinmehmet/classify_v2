"""
@author: funda
@author: uğuray

"""

import numpy as np
import pandas as pd

# from intent_keywords import *
from .preprocess import Preprocessor
from .clana_ import Clana
from .param_config import VECT_PARAMS, VECTORIZER_FOLDER, \
                            MODEL_SAVE, PREP_PARAMS
from .vectorization import train_vectorizer, predict_vectorizer
# from grid_search import find_estimator
from .performance_results import get_performance_results
from sklearn.model_selection import train_test_split,  cross_val_predict
from sklearn import linear_model
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
# from sklearn.metrics import ConfusionMatrixDisplay 
from sklearn.metrics import confusion_matrix


def Train(module_id, texts, labels, language='tr', params=None):
    '''
    Train the model.
    
    Parameters
    ----------
    df : DataFrame[Text, Label] or str
    name : str

    Returns
    -------
    model : object
    cls_reports : dict

    '''

    # preprocess steps
    # steps = pd.DataFrame(PREP_PARAMS.items(), columns = ["Step",".."])
    # print(tabulate(steps, headers=["Preprocess Step", "Default Value"],  
    #                                 tablefmt="fancy_grid", showindex=False))
    
    # preprocess
    table =  {'text':texts, 'label':labels}
    df = pd.DataFrame(table)
    name = module_id
    #update PREP_PARAMS via PREP_box

    #update params into PREP_PARAMS
    #YAZILACAK
    #PREP_PARAMS = params

    data = Preprocessor(df, params)
    labels = data['label']
    
    # train, test split
    train, test, train_labels, test_labels = train_test_split(data, 
                                                              labels, 
                                                              stratify=labels,
                                                              test_size=params["SPLIT_RATIO_CV"]["TEST_RATIO"],
                                                              random_state=42)
    '''
    # train and test class distributions
    train_df = pd.DataFrame({'text':train['after'].tolist(), 'label':train_labels})
    train_class_dist = train_df['label'].value_counts().rename_axis('class').to_frame('counts')
    test_df = pd.DataFrame({'text':test['after'].tolist() , 'label':test_labels})
    test_class_dist = test_df['label'].value_counts().rename_axis('class').to_frame('counts')
    class_dist = pd.merge(train_class_dist, test_class_dist, on='class')
    print(tabulate(class_dist, headers=["Class", "Train", "Test"], tablefmt="fancy_grid"))
    '''

    # vectorization
    train_tfidf, test_tfidf, vectorizer = train_vectorizer(train = train,
                                                           test = test,
                                                           data = [])
    
    # grid-search
    # estimator, score = find_estimator(train_tfidf, train_labels)
    
    # classification
    classifier = linear_model.SGDClassifier(alpha=0.001, loss='hinge')
    model = CalibratedClassifierCV(base_estimator=classifier, method='sigmoid')
    model.fit(train_tfidf, train_labels)
    
    # prediction 
    ypred = model.predict(test_tfidf)
    yscore = model.predict_proba(test_tfidf)
    
    # performance metrics
    results_dict = get_performance_results(test_labels, ypred, yscore)
    
    # extract labels from clana

    clana_labels = Clana(test_labels, name, results_dict)

    # confusion matrix
    cm = confusion_matrix(test_labels, ypred, clana_labels) 
    # conf_matrix = ConfusionMatrixDisplay(confusion_matrix=cm, 
    #                                      display_labels=clana_labels)
    # conf_matrix.plot(xticks_rotation='vertical')
    # conf_matrix.figure_.savefig(name+'.png')
    # plt.show()

   # saving model and vectorizer
   #  if MODEL_SAVE:
   #      model_path = 'model_('+str(name)+').pkl'
   #      vectorizer_name = 'vectorizer_('+str(name)+').pkl'

    from classify_v2.app.service import Root
    root = Root({})

    root['module_id'] = module_id
    root['parent_category_id'] = 0
    root['name']= name
    root['description'] = "root"
    root['statu'] = 'D'
    root['sample_number'] = results_dict['data_summary']['number_of_instance']
    root['accuracy'] = results_dict['accuracy']['overall']
    root['f1_score'] = results_dict['fscores']['overall']['macro']
    root['precision'] = results_dict['precision']['overall']['macro']
    root['recall'] = results_dict['recall']['overall']['macro']
    root['truepositive'] = results_dict['conf_matrix_val']['overall']['TP']
    root['truenegative'] = results_dict['conf_matrix_val']['overall']['TN']
    root['falsepositive'] = results_dict['conf_matrix_val']['overall']['FP']
    root['falsenegative'] = results_dict['conf_matrix_val']['overall']['FN']
    root['roc_curve'] = results_dict['ROC_curves']['overall']
    root['pr_curve'] = results_dict['PR_curves']['overall']
    root['gmean'] = results_dict['gmean']['overall']['macro']
    root['auc_score'] = results_dict['ROC_values']['average_auc']
    root['auprc_score'] = results_dict['average_precision_scores']['micro']
    root['roc'] = results_dict['ROC_values']['roc_auc']['micro']
    root['confusion_matrix'] = results_dict['confusion_matrix'].tolist()
    root['true_positive_rate'] = list(results_dict['ROC_values']['tpr']['micro'].flatten())
    root['false_positive_rate'] = list(results_dict['ROC_values']['fpr']['micro'].flatten())
    root['precision_recall_curve'] = results_dict['precision_recall_curve']
    root['keywords'] = ["what", "the", "hell"]
    
    # root["thresholds"] =  results_dict['ROC_values']['thresholds'].items()
 
    root['confusion_matrix_classes'] = results_dict['data_summary']['classes']
    
#    Categories
    classes =  results_dict['data_summary']['classes']
    
    root['categoryList'] = []
    index = -1
    for c in classes:
        index += 1
        category = Root({})
        category.sample_number = root['sample_number']
        category.name = c
        # if category.description is None or category.description.strip() == '':
        #     category.description = c
        # else:
        category.description = c
        category.accuracy = results_dict['accuracy']['classes'][c]
        category.f1_score = results_dict['fscores']['classes'][c]
        category.precision = results_dict['precision']['classes'][c]
        category.recall =results_dict['recall']['classes'][c]
        category.gmean =  results_dict['gmean']['classes'][c]
        category.truepositive =  results_dict['TPs_per_class'][index]
        category.truenegative =  results_dict['TNs_per_class'][index]
        category.falsepositive = results_dict['FPs_per_class'][index]
        category.falsenegative = results_dict['FNs_per_class'][index]
        category.keywords = ["what", "the", "hell"]
        category.confusion_matrix = results_dict['confusion_matrix_per_class'][index].tolist()
        category['confusion_matrix_classes'] = results_dict['data_summary']['classes']
        # category.thresholds = category.thresholds # edit
        category.true_positive_rate =  list(results_dict['ROC_values']['fpr'][c].flatten())
        category.false_positive_rate = list(results_dict['ROC_values']['fpr'][c].flatten())
        category['auc_score'] = results_dict['ROC_values']['average_auc']
        category['auprc_score'] = results_dict['average_precision_scores'][index] # edit
        category['roc'] = results_dict['ROC_values']['roc_auc'][c]
        category.precision_recall_curve = root['precision_recall_curve'] # edit
        category.roc_curve = results_dict['roc_curves']['classes'][c]
        category.pr_curve = results_dict['pr_curves']['classes'][c]
        category.statu = "A"
        category.module_id = module_id
        root['categoryList'].append(category)

    return model, vectorizer, root


def Train_CV(module_id, texts, labels, language='tr', params=None):
    '''
    Train the model.

    Parameters
    ----------
    df : DataFrame[Text, Label] or str
    name : str

    Returns
    -------
    model : object
    cls_reports : dict

    '''
    
    # preprocess

    table = {'text': texts, 'label': labels}
    df = pd.DataFrame(table)
    name = module_id
    # update PREP_PARAMS via PREP_box

    # update params into PREP_PARAMS
    # YAZILACAK
    # PREP_PARAMS = params

    data = Preprocessor(df, params["PREP_PARAMS"])
    labels = data['label']
    
    # vectorization
    train_tfidf, vectorizer = train_vectorizer(data = data)
    
    # classification
    classifier = linear_model.SGDClassifier(alpha = 0.001,
                                            loss = 'hinge',
                                            random_state = 0)
    model = CalibratedClassifierCV(base_estimator = classifier, method = 'sigmoid')
    model.fit(train_tfidf, labels)
    
    # prediction
    yscore = cross_val_predict(model,
                               train_tfidf,
                               labels,
                               method = "predict_proba",
                               cv = params['SPLIT_RATIO_CV']['CV'])
    
    # ypred
    label_names = sorted(list(set(labels)))
    ypred_ = yscore.argmax(axis = 1)
    ypred = np.array([label_names[i] for i in ypred_])
    
    # performance metrics
    results_dict = get_performance_results(labels, ypred, yscore)
    
    # extract labels from clana
    clana_labels = Clana(labels, name, results_dict)
    
    # confusion matrix
    cm = confusion_matrix(labels, ypred, clana_labels)

    from classify_v2.app.service import Root
    root = Root({})

    root['module_id'] = module_id
    root['parent_category_id'] = 0
    root['name'] = name
    root['description'] = "root"
    root['statu'] = 'D'
    root['sample_number'] = results_dict['data_summary']['number_of_instance']
    root['accuracy'] = results_dict['accuracy']['overall']
    root['f1_score'] = results_dict['fscores']['overall']['macro']
    root['precision'] = results_dict['precision']['overall']['macro']
    root['recall'] = results_dict['recall']['overall']['macro']
    root['truepositive'] = results_dict['conf_matrix_val']['overall']['TP']
    root['truenegative'] = results_dict['conf_matrix_val']['overall']['TN']
    root['falsepositive'] = results_dict['conf_matrix_val']['overall']['FP']
    root['falsenegative'] = results_dict['conf_matrix_val']['overall']['FN']
    root['roc_curve'] = results_dict['ROC_curves']['overall']
    root['pr_curve'] = results_dict['PR_curves']['overall']
    root['gmean'] = results_dict['gmean']['overall']['macro']
    root['auc_score'] = results_dict['ROC_values']['average_auc']
    root['auprc_score'] = results_dict['average_precision_scores']['micro']
    root['roc'] = results_dict['ROC_values']['roc_auc']['micro']
    root['confusion_matrix'] = results_dict['confusion_matrix'].tolist()
    root['true_positive_rate'] = list(results_dict['ROC_values']['tpr']['micro'].flatten())
    root['false_positive_rate'] = list(results_dict['ROC_values']['fpr']['micro'].flatten())
    root['precision_recall_curve'] = results_dict['precision_recall_curve']
    root['keywords'] = ["what", "the", "hell"]

    # root["thresholds"] =  results_dict['ROC_values']['thresholds'].items()

    root['confusion_matrix_classes'] = results_dict['data_summary']['classes']

    #    Categories
    classes = results_dict['data_summary']['classes']

    root['categoryList'] = []
    index = -1
    for c in classes:
        index += 1
        category = Root({})
        category.sample_number = root['sample_number']
        category.name = c
        # if category.description is None or category.description.strip() == '':
        #     category.description = c
        # else:
        category.description = c
        category.accuracy = results_dict['accuracy']['classes'][c]
        category.f1_score = results_dict['fscores']['classes'][c]
        category.precision = results_dict['precision']['classes'][c]
        category.recall = results_dict['recall']['classes'][c]
        category.gmean = results_dict['gmean']['classes'][c]
        category.truepositive = results_dict['TPs_per_class'][index]
        category.truenegative = results_dict['TNs_per_class'][index]
        category.falsepositive = results_dict['FPs_per_class'][index]
        category.falsenegative = results_dict['FNs_per_class'][index]
        category.keywords = ["what", "the", "hell"]
        category.confusion_matrix = results_dict['confusion_matrix_per_class'][index].tolist()
        category['confusion_matrix_classes'] = results_dict['data_summary']['classes']
        # category.thresholds = category.thresholds # edit
        category.true_positive_rate = list(results_dict['ROC_values']['fpr'][c].flatten())
        category.false_positive_rate = list(results_dict['ROC_values']['fpr'][c].flatten())
        category['auc_score'] = results_dict['ROC_values']['average_auc']
        category['auprc_score'] = results_dict['average_precision_scores'][index]  # edit
        category['roc'] = results_dict['ROC_values']['roc_auc'][c]
        category.precision_recall_curve = root['precision_recall_curve']  # edit
        category.roc_curve = results_dict['roc_curves']['classes'][c]
        category.pr_curve = results_dict['pr_curves']['classes'][c]
        category.statu = "A"
        category.module_id = module_id
        root['categoryList'].append(category)

    return model, vectorizer, root


def Predict(test_data, vectorizer_file, model_file, params):
    '''
    Use to make predictions on new data instances.

    Parameters
    ----------
    test_data : DataFrame[Text] or str
    vectorizer_file : str
    model_file : str

    Returns
    -------
    predicted_labels : list
    cats_probs : list[Tuple]

    '''

    # preprocess
    data = Preprocessor(test_data, params)
    
    # get vectorizer and model
    test_tfidf = predict_vectorizer(data, vectorizer_file)
    classifier = model_file
    
    # prediction
    # t0 = time()
    prediction_probabilities = classifier.predict_proba(test_tfidf)
    # t1 = time() 
    # print("\nPrediction took {} sec.\n".format(t1 - t0))
    
    # get class prediction and probabilities for each class 
    categories = classifier.classes_
    if isinstance(test_data, str): data = [data]
    
    predicted_labels = []
    for i in range(len(data)):
        label_index = np.argmax(prediction_probabilities[i])
        label = categories[label_index]
        predicted_labels.append(label)
        cats_probs = {label : round(probability,2) for label, probability in \
                      zip(categories, prediction_probabilities[i])}

    # sort dictionary according to probabilities
    # cats_probs = sorted(cats_probs.items(), key=lambda item: item[1], reverse=True)
    cats_probs= {k: v for k, v in sorted(cats_probs.items(), key = lambda item: item[1],  reverse=True)}
    # print predicted class
    # pred_cls = pd.DataFrame({"text":test_data, "predicted_cls":predicted_labels})
    # headers = ['Test', 'Predicted class']
    # print(tabulate(pred_cls, headers=headers, tablefmt="fancy_grid", showindex=False))
    
    # print class and probabilities
    # headers = ['Class', 'Probability']
    # print(tabulate(cats_probs, headers=headers, tablefmt="fancy_grid"))

    return  predicted_labels, cats_probs

    
  

