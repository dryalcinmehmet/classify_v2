"""
@author: funda
@author: uğuray

"""

# from params_config import grid_params
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier


def find_estimator(matrix, labels):
    '''
    Model parameters optimization with GridSearchCV

    Parameters
    ----------
    matrix : np.float object 
    labels : pd.series 

    Returns
    -------
    best_estimator_ : object 
    best_score_ : float

    '''

    # parameters for GridSearchCV
    # cv = grid_params['cv']
    cv = 10
    
    # paramaters for SGDClassifier
    params = {
        "loss" : ['hinge', 'log', 'squared_hinge', 'modified_huber'],
        "alpha" : [0.0001, 0.001, 0.01, 0.1],
        "penalty" : ["l1", "l2"]
    }
    
    model = SGDClassifier(max_iter=1000)
    clf = GridSearchCV(model, param_grid=params, scoring='accuracy', cv=cv)
    clf.fit(matrix, labels)
    
    # print(clf.best_score_)
    # print(clf.best_estimator_)
    # print(clf.cv_results_)
    
    return clf.best_estimator_, clf.best_score_
