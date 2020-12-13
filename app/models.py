from app import db
import json
from sqlalchemy.ext import mutable
try:
    from sqlalchemy import ARRAY
except ImportError:
    from sqlalchemy.dialects.postgresql import ARRAY

class JsonEncodedDict(db.TypeDecorator):
    impl = db.Text
    
    def process_bind_param(self, value, dialect):
        if value is None:
            return '{}'
        else:
            return json.dumps(value)
    
    def process_result_value(self, value, dialect):
        if value is None:
            return {}
        else:
            return json.loads(value)


mutable.MutableDict.associate_with(JsonEncodedDict)


class Module(db.Model):
    __tablename__ = 'cog_module'

    id =  db.Column(db.Integer, primary_key = True)
    user_id = db.Column(db.Integer)
    name = db.Column(db.String)
    description = db.Column(db.String)
    statu = db.Column(db.String)
    create_date = db.Column(db.DateTime)
    model_name = db.Column(db.String)
    stopword = db.Column(db.Boolean)
    stemming = db.Column(db.Boolean)
    remove_numbers = db.Column(db.Boolean)
    deasciify = db.Column(db.Boolean)
    remove_punkt = db.Column(db.Boolean)
    lowercase = db.Column(db.Boolean)
    spellcheck = db.Column(db.Boolean)
    wordngram_start = db.Column(db.Integer)
    wordngram_end = db.Column(db.Integer)
    accuracy = db.Column(db.Integer)
    phrase_match = db.Column(db.Boolean)
    shuffle_phrase_match = db.Column(db.Boolean)
    word_match = db.Column(db.Boolean)
    partial_match_rate = db.Column(db.Float)
    attributes = db.Column(db.Text)
    target = db.Column(db.Text)
    auto_train_when_data_change = db.Column(db.Boolean)
    return_unknown_category = db.Column(db.Boolean)
    language_id =  db.Column(db.Integer, db.ForeignKey('cog_language_parameter.id'))
    module_type_id = db.Column(db.Integer)
    attribute_limit = db.Column(db.Integer)
    data_limit = db.Column(db.Integer)
    language = db.relationship('LanguageParameter')


    def __repr__(self):
        return '<Module {}>'.format(self.id)

class ModuleCategory(db.Model):
    __tablename__ = 'cog_module_category'

    id =  db.Column(db.Integer, primary_key = True)
    sample_number =  db.Column(db.Integer)
    precision =  db.Column(db.Float)
    recall =  db.Column(db.Float)
    accuracy =  db.Column(db.Float)
    f1_score =  db.Column(db.Float)
    truepositive =  db.Column(db.Integer)
    truenegative =  db.Column(db.Integer)
    falsepositive =  db.Column(db.Integer)
    falsenegative =  db.Column(db.Integer)
    gmean =  db.Column(db.Float)
    statu =  db.Column(db.String)
    name =  db.Column(db.String)
    keywords =  db.Column(db.String)
    description =  db.Column(db.String)
    confusion_matrix = db.Column(ARRAY(db.Integer))
    confusion_matrix_classes =  db.Column(ARRAY(db.String))
    auc_score =  db.Column(db.Float)
    auprc_score =  db.Column(db.Float)
    true_positive_rate =  db.Column(ARRAY(db.Float))
    false_positive_rate =  db.Column(ARRAY(db.Float))
    thresholds =  db.Column(ARRAY(db.Float))
    pr =  db.Column(ARRAY(db.Float))
    rec =  db.Column(ARRAY(db.Float))
    thr =  db.Column(ARRAY(db.Float))
    precision_recall_curve = db.Column(ARRAY(db.Float))
    roc =  db.Column(db.Float)
    pr_curve = db.Column(db.Text)
    roc_curve = db.Column(db.Text)
    module_id =  db.Column(db.Integer, db.ForeignKey('cog_module.id'))
    parent_category_id =  db.Column(db.Integer)
    train_log_id =   db.Column(db.Integer, db.ForeignKey('dashboard_trainlog.id'))


class LanguageParameter(db.Model):
    __tablename__ = 'cog_language_parameter'

    id =  db.Column(db.Integer, primary_key = True)
    name = db.Column(db.String)
    code =  db.Column(db.String)
    statu = db.Column(db.String)

class ModuleData(db.Model):
    __tablename__ = 'cog_module_data'

    id =  db.Column(db.Integer, primary_key = True)
    text =  db.Column(db.String)
    label =  db.Column(db.String)
    created_at = db.Column(db.DateTime)
    updated_at = db.Column(db.DateTime)
    module_id =  db.Column(db.Integer, db.ForeignKey('cog_module.id'))

    def __repr__(self):
        return '<Module %r>' % self.id


class TrainLog(db.Model):
    __tablename__ = 'dashboard_trainlog'

    id =  db.Column(db.Integer, primary_key = True)
    module_id =  db.Column(db.Integer, db.ForeignKey('cog_module.id'))
    started_at = db.Column(db.DateTime)
    finished_at = db.Column(db.DateTime)
    is_running = db.Column(db.Boolean)
    data_count =  db.Column(db.Integer)
    used_stopwords = db.Column(db.Boolean)
    used_stemming = db.Column(db.Boolean)
    used_remove_numbers = db.Column(db.Boolean)
    used_deasciify = db.Column(db.Boolean)
    used_remove_punkt = db.Column(db.Boolean)
    used_lowercase = db.Column(db.Boolean)
    used_spellcheck = db.Column(db.Boolean)

    celery_task_id =  db.Column(db.String)
    celery_task_status =  db.Column(db.String)
    version = db.Column(db.Integer, default = 2)


class TrainTestLog(db.Model):
    __tablename__ = 'dashboard_traintestlog'
    
    id =  db.Column(db.Integer, primary_key = True)
    module_id =  db.Column(db.Integer, db.ForeignKey('cog_module.id'))
    started_at = db.Column(db.DateTime)
    finished_at = db.Column(db.DateTime)
    is_running = db.Column(db.Boolean)
    data_count =  db.Column(db.Integer)
    used_stopwords = db.Column(db.Boolean)
    used_stemming = db.Column(db.Boolean)
    used_remove_numbers = db.Column(db.Boolean)
    used_deasciify = db.Column(db.Boolean)
    used_remove_punkt = db.Column(db.Boolean)
    used_lowercase = db.Column(db.Boolean)
    used_spellcheck = db.Column(db.Boolean)
    celery_task_id = db.Column(db.String)
    version = db.Column(db.Integer, default = 2)

    accuracy = db.Column(db.Float)
    f1_score = db.Column(db.Float)
    precision = db.Column(db.Float)
    recall = db.Column(db.Float)
    auc = db.Column(db.Float)

    def __str__(self):
        import datetime
        status = " Running" if self.is_running else " Finished"
        if not self.module.accuracy == 0:
            return datetime.strftime(
                self.started_at,
                "%Y-%m-%d %H:%M") + status + \
                " - Accuracy: " + str(self.module.accuracy) + "%"
        else:
            return datetime.strftime(
                self.started_at,
                "%Y-%m-%d %H:%M") + status


class ClassifyTrain(db.Model):
    __tablename__ = 'user_classifytrain'
    id = db.Column(db.Integer, primary_key = True)
    module_id = db.Column(db.Integer, db.ForeignKey('cog_module.id'))
    is_finished = db.Column(db.Boolean, default=False)
    started_at =  db.Column(db.DateTime)
    finished_at = db.Column(db.DateTime)


class ClassifyFeedback(db.Model):
    __tablename__ = 'user_classifyfeedback'
    id = db.Column(db.Integer, primary_key = True)
    module_id = db.Column(db.Integer, db.ForeignKey('cog_module.id'))
    log_id = db.Column(db.ForeignKey('api_cognituslog.id'),  unique=True)
    text = db.Column(db.Text)
    predicted_category = db.Column(db.String)
    true_category =db.Column(db.String)
    label = db.Column(db.Boolean, default=True)
    is_active = db.Column(db.Boolean, default=True)
    source = db.Column(db.String)
    created_at = db.Column(db.DateTime)
    updated_at = db.Column(db.DateTime )
    request_id = db.Column(db.String)

class CognitusLog(db.Model):
    __tablename__ = 'api_cognituslog'
    id = db.Column(db.Integer, primary_key = True)