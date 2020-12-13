import numpy
from app import db
from app import celery
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from app.models import TrainLog, TrainTestLog
from .service import ClassifyMiddleware
from app.models import (ModuleCategory,  ModuleData,
                        Module, LanguageParameter, ClassifyTrain,
                        ClassifyFeedback)

from psycopg2.extensions import register_adapter, AsIs
def addapt_numpy_float64(numpy_float64):
    return AsIs(numpy_float64)
def addapt_numpy_int64(numpy_int64):
    return AsIs(numpy_int64)
register_adapter(numpy.float64, addapt_numpy_float64)
register_adapter(numpy.int64, addapt_numpy_int64)

@celery.task(bind = True)
def predict_service(self, module_id, texts, request_id, labels=None, language=None):
    import json
    from .service import Root
    obj = ClassifyMiddleware(module_id, texts, labels=None, language=None)
    response = obj.predict()
    result = json.dumps(response, skipkeys=False, ensure_ascii=False)
    result_serialized = json.loads(result)
    res = Root({})
    res['results'] = []
    for k,v in result_serialized.items():
        res['results'].append( {"category" : k, "probability":v} )
    insert_classify_v2_feedback(module_id, texts, "", "", request_id)
    return res

@celery.task(bind = True)
def train_service(self, module_id, texts, labels, language, split, initial_params, is_test= False):
    celery_task_id = self.request.id
    obj = ClassifyMiddleware(module_id, texts, labels, language, celery_task_id, initial_params= initial_params, is_test = is_test)
    learning_model, vectorizer, root = obj.train(split)
    if is_test:
        insert_train_test_log(
                module_id = module_id,
                data_count = root.sample_number,
                celery_task_id = celery_task_id,
                stopwords = initial_params['stopwords'],
                stemming = initial_params['stemming'],
                remove_numbers = initial_params['remove_numbers'],
                deasciify = initial_params['deasciify'],
                remove_punkt = initial_params['remove_punkt'],
                lowercase = initial_params['lowercase'],
                accuracy= int(root.accuracy*100),
                f1_score = int(root.f1_score*100),
                precision = int(root.precision*100),
                recall= int(root.recall*100),
                auc = int(root.auc_score*100)
        )
        
    else:
        insert_train_log(
                module_id = module_id,
                data_count = root.sample_number,
                celery_task_id = celery_task_id,
                stopwords = initial_params['stopwords'],
                stemming = initial_params['stemming'],
                remove_numbers = initial_params['remove_numbers'],
                deasciify = initial_params['deasciify'],
                remove_punkt = initial_params['remove_punkt'],
                lowercase = initial_params['lowercase']
        )
        train_log_id = get_module_train_log(module_id)
        root['train_log_id'] = train_log_id
        i = insert_module_category(root)
        insert_classify_v2_train(module_id)
        for category in root.categoryList:
            insert_module_category_list(category, train_log_id)
    return 'Success'

def insert_module_category(root):
    cog_module_category = ModuleCategory(
                            sample_number = root.sample_number,
                            precision = root.precision,
                            recall = root.recall,
                            accuracy = root.accuracy,
                            f1_score = root.f1_score,
                            truepositive = root.truepositive,
                            truenegative = root.truenegative,
                            falsepositive = root.falsepositive,
                            falsenegative = root.falsenegative,
                            gmean = root.gmean,
                            statu = root.statu,
                            name = root.name,
                            keywords = root.keywords,
                            description = root.description,
                            confusion_matrix = root.confusion_matrix,
                            confusion_matrix_classes = root.confusion_matrix_classes,
                            auc_score = root.auc_score,
                            auprc_score = root.auprc_score,
                            true_positive_rate = root.true_positive_rate,
                            false_positive_rate = root.false_positive_rate,
                            # thresholds = root.thresholds,
                            # pr = None,
                            # rec = None
                            # thr = None,
                            precision_recall_curve = root.precision_recall_curve,
                            roc = root.roc,
                            pr_curve = root.pr_curve,
                            roc_curve = root.roc_curve,
                            module_id = root.module_id,
                            # parent_category_id = root.parent_category_id,
                            train_log_id = root.train_log_id
    )
    db.session.add(cog_module_category)
    try:
        db.session.commit()
    except:
        db.session.rollback()
        raise
    # finally:
    #     db.session.close()
    # return True


def insert_module_category_list(category, train_log_id):
    cog_module_category = ModuleCategory(
                            sample_number = category.sample_number,
                            precision = category.precision,
                            recall = category.recall,
                            accuracy = category.accuracy,
                            f1_score = category.f1_score,
                            truepositive = category.truepositive,
                            truenegative = category.truenegative,
                            falsepositive = category.falsepositive,
                            falsenegative = category.falsenegative,
                            gmean = category.gmean,
                            statu = category.statu,
                            name = category.name,
                            keywords = category.keywords,
                            description = category.description,
                            confusion_matrix = category.confusion_matrix,
                            confusion_matrix_classes = category.confusion_matrix_classes,
                            auc_score = category.auc_score,
                            auprc_score = category.auprc_score,
                            true_positive_rate = category.true_positive_rate,
                            false_positive_rate = category.false_positive_rate,
                            #thresholds = category.thresholds,
                            # pr = None,
                            # rec = None
                            # thr = None,
                            precision_recall_curve = category.precision_recall_curve,
                            roc = category.roc,
                            pr_curve = category.pr_curve,
                            roc_curve = category.roc_curve,
                            module_id = category.module_id,
                            # parent_category_id = category.parent_category_id,
                            train_log_id = train_log_id
    )
    db.session.add(cog_module_category)
    try:
        db.session.commit()
    except:
        db.session.rollback()
        raise
    finally:
        db.session.close()
    return True

def select_module(module_id):
    cog_module_data = ModuleData.query.filter_by(module_id = module_id).all()
    return cog_module_data

def insert_train_log(module_id, data_count, celery_task_id,  stopwords,stemming, remove_numbers,
                     deasciify, remove_punkt, lowercase):
    now = datetime.today()
    train_log = TrainLog(module_id=module_id, started_at = now,
                         is_running=True, data_count=data_count,
                         used_stopwords=stopwords,
                         used_stemming = stemming,
                         used_remove_numbers = remove_numbers,
                         used_deasciify = deasciify,
                         used_remove_punkt = remove_punkt,
                         used_lowercase = lowercase,
                         celery_task_id=celery_task_id,
                         version = 2)
    db.session.add(train_log)
    try:
        db.session.commit()
    except:
        db.session.rollback()
        raise

def insert_train_test_log(module_id, data_count, celery_task_id,  stopwords,stemming, remove_numbers,
                     deasciify, remove_punkt, lowercase,accuracy, f1_score, precision, recall, auc):
    now = datetime.today()
    train_log = TrainTestLog(module_id=module_id, started_at = now,
                         is_running=True, data_count=data_count,
                         used_stopwords=stopwords,
                         used_stemming = stemming,
                         used_remove_numbers = remove_numbers,
                         used_deasciify = deasciify,
                         used_remove_punkt = remove_punkt,
                         used_lowercase = lowercase,
                         celery_task_id=celery_task_id,
                         version = 2,
                         accuracy = accuracy,
                         f1_score = f1_score,
                         precision =precision,
                         recall = recall,
                         auc = auc)
    db.session.add(train_log)
    try:
        db.session.commit()
    except:
        db.session.rollback()
        raise
    
def update_train_log(train_log_id):
    now = datetime.today()
    train_log = TrainLog.query.filter_by(id=train_log_id).first()
    train_log.is_running = False
    train_log.finished_at = now
    db.session.add(train_log)
    try:
        db.session.commit()
    except:
        db.session.rollback()
        raise

def insert_classify_v2_train(module_id):
    now = datetime.today()
    ctrain_log = ClassifyTrain(module_id=module_id, started_at = now,
                                  finished_at = now, is_finished=True)
    db.session.add(ctrain_log)
    try:
        db.session.commit()
    except:
        db.session.rollback()
        raise

def insert_classify_v2_feedback(module_id,text,predicted_category,true_category, request_id):
    now = datetime.today()
    feedback = ClassifyFeedback(module_id=module_id,
                                        text = "",
                                        predicted_category = "",
                                        true_category = "",
                                        label = False,
                                        is_active = False,
                                        source = "",
                                        created_at = now,
                                        updated_at = now,
                                        request_id = request_id
                                    )
    db.session.add(feedback)
    try:
        db.session.commit()
    except:
        db.session.rollback()
        raise
    

def get_test_ratio(module_id):
    module = Module.query.filter_by(id=module_id).first()
    return module.test_ratio

def get_cv(module_id):
    module = Module.query.filter_by(id=module_id).first()
    return module.cv

def get_split(module_id):
    module = Module.query.filter_by(id=module_id).first()
    return module.split

def get_stopwords(module_id):
    module = Module.query.filter_by(id=module_id).first()
    return module.stopword

def get_module_language(language_id):
    module = LanguageParameter.query.filter_by(id = language_id).first()
    return module.code

def get_module_train_log(module_id):
    module = TrainLog.query.filter_by(module_id=module_id).first()
    return module.id

def get_module(module_id):
    module = Module.query.filter_by(id=module_id).first()
    return module
