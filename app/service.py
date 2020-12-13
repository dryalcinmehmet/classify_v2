from __future__ import absolute_import
import os
import sys
import boto3
import pickle
import redis
from datetime import datetime
from celery import Celery
from modules import classify
from modules import param_config
from dotenv import load_dotenv

load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME", "cognitus-fs")
DYNAMIC_CLASSIFY_PATH = os.getenv("DYNAMIC_CLASSIFY_PATH", "/srv/dynamic_classify_models")
AWS_REGION_NAME = os.getenv('AWS_REGION_NAME')

redis_config = {}
if os.environ.get("REDIS_PASSWORD"):
	redis_config["password"] = os.environ.get("REDIS_PASSWORD")
redis_client = redis.Redis(os.environ.get("REDIS_HOST"), **redis_config)
if os.environ.get("REDIS_PASSWORD"):
	REDIS_URL = 'redis://:{}@{}:{}/{}'.format(
			os.environ.get("REDIS_PASSWORD"),
			os.environ.get("REDIS_HOST"),
			os.environ.get("REDIS_PORT"),
			os.environ.get("REDIS_DB")
	)
else:
	REDIS_URL = 'redis://{}:{}/{}'.format(
			os.environ.get("REDIS_HOST", 'redis'),
			os.environ.get('REDIS_PORT'),
			os.environ.get('REDIS_DB')
	)

AWS_S3_ENDPOINT_URL = os.getenv("AWS_S3_ENDPOINT_URL")
AWS_BUCKET_NAME = os.getenv("AWS_STORAGE_BUCKET_NAME")
s3_client_kwargs = {"aws_access_key_id": AWS_ACCESS_KEY_ID,
                    "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
                    "endpoint_url": AWS_S3_ENDPOINT_URL
                    }


sys.path.append("")


class ClassifyMiddleware:
	
	def __init__(self, module_id = None, texts = None, labels = None, language = None, celery_task_id = None,
	             initial_params = None, is_test = False):
		self.module_id = module_id
		self.texts = texts
		self.labels = labels
		self.language = language
		self.modelrootfolder = "models"
		self.modelname = param_config.model_name
		self.model_folder = str(self.module_id) + "_model"
		self.vectrootfolder = "models"
		self.vectname = param_config.vectorizer_name
		self.dump_to_s3 = True
		self.PARAMS = self.update_params()
		self.model_file, self.vect_file = self.files()
		self.learning_model = ""
		self.vectorizer = ""
		self.celery_task_id = celery_task_id
		self.is_test = is_test
	
	def files(self):
		try:
			self.model_file, self.vect_file \
				= self.load_files(self.modelrootfolder)
		except:
			self.model_file = None
			self.vect_file = None
		return self.model_file, self.vect_file
	
	def train(self, split):
		# from celery.contrib import rdb; rdb.set_trace()
		#train a classification model
		if split != 100:
			
			self.learning_model, self.vectorizer, self.root = classify.Train(self.module_id, self.texts, self.labels,params = self.PARAMS)
			if not self.is_test:
				self.execute()
		else:
			# train a model with all data
			self.learning_model, self.vectorizer, self.root = classify.Train_CV(self.module_id, self.texts, self.labels,
			                                                                    params = self.PARAMS)
			if not self.is_test:
				self.execute()
				
		return self.learning_model, self.vectorizer, self.root
	
	def execute(self):
		model = self.dump_file(self.learning_model,
		                       self.modelrootfolder,
		                       self.modelname,
		                       dump_to_s3 = self.dump_to_s3,
		                       vectorizer = self.vectorizer,
		                       vectrootfolder = self.vectrootfolder,
		                       vectname = self.vectname)
		return model
	
	def predict(self):
		model, vectorizer = self.load_files(picklefolder = self.modelrootfolder)
		predicted_labels, prediction_map = classify.Predict(self.texts, vectorizer, model, self.PARAMS)
		results = prediction_map
		return results
	
	def ensure_dir(self, folder):
		if not os.path.exists(folder):
			os.makedirs(folder)
		return folder
	
	def dump_file(self, model,
	              modelfolder,
	              modelname, dump_to_s3 = False, vectorizer = None, vectrootfolder = None,
	              vectname = None):
		recordfolder = os.path.join(modelfolder, self.model_folder)
		self.ensure_dir(recordfolder)
		modelpath = os.path.join(recordfolder, modelname)
		recordfoldervect = os.path.join(vectrootfolder, vectname)
		self.ensure_dir(recordfoldervect)
		vectpath = os.path.join(recordfolder, vectname)
		self.dump_object(vectorizer, vectpath, dump_to_s3 = dump_to_s3)
		self.dump_object(model, modelpath, dump_to_s3 = dump_to_s3)
		redis_client.set("%s_trained" % recordfolder, 1)
		redis_client.set(recordfolder + "trained_at",
		                 pickle.dumps(datetime.now().timestamp()))
		return recordfolder
	
	def dump_object(self, obj, model_path, dump_to_s3 = False):
		if dump_to_s3:
			client = boto3.client('s3', **s3_client_kwargs)
			client.put_object(Body = pickle.dumps(obj), Bucket = os.environ.get("AWS_BUCKET_NAME"),
			                  Key = model_path)
			redis_client.delete(model_path)
			redis_client.set(model_path, pickle.dumps(obj))
		else:
			with open(model_path, "wb") as ff:
				pickle.dump(obj, ff)
			redis_client.delete(model_path)
			redis_client.set(model_path, pickle.dumps(obj))
	
	def load_files(self, picklefolder = None, load_from_file = False):
		if picklefolder == None:
			picklefolder = self.modelrootfolder
		modelpath = os.path.join(picklefolder + '/{}_model'.format(self.module_id), param_config.model_name)
		vectorizer = os.path.join(picklefolder + '/{}_model'.format(self.module_id), param_config.vectorizer_name)
		if self.model_file:
			if not redis_client.exists(picklefolder + "trained_at"):
				return self.model_file, self.vect_file
			trained_at = pickle.loads(redis_client.get(
					picklefolder + "trained_at"))
			if trained_at <= self.load_time:
				return self.model_file, self.vect_file
		
		model = self.load_object(modelpath, load_from_file = load_from_file)
		vectorizer = self.load_object(vectorizer, load_from_file = load_from_file)
		self.model_file = model
		self.vect_file = vectorizer
		self.load_time = datetime.now().timestamp()
		return model, vectorizer
	
	def load_object(self, model_path, load_from_file = False):
		if not load_from_file:
			obj = redis_client.get(model_path)
			if obj:
				return pickle.loads(obj)
			client = boto3.client('s3', **s3_client_kwargs)
			try:
				obj = client.get_object(Bucket = os.environ.get("AWS_BUCKET_NAME"), Key = model_path)
				model = obj['Body'].read()
				redis_client.set(model_path, model)
				return pickle.loads(model)
			except:
				return
		else:
			obj = redis_client.get(model_path)
			if obj:
				return pickle.loads(obj)
			if not os.path.exists(model_path):
				return
			with open(model_path, "rb") as ff:
				obj = pickle.load(ff)
				redis_client.set(model_path, pickle.dumps(obj))
				return obj
	
	def update_params(self):

		self.PARAMS = {"PREP_PARAMS" : param_config.PREP_PARAMS,
		               "SPLIT_RATIO_CV" : param_config.SPLIT_RATIO_CV}
		# params içinden PREP_PARAMSI GÜNCELLE
		# UPDATE AFTER DB CHANGES
		try:
			self.PARAMS['PREP_PARAMS']["stopwords"] = self.initial_params["stopwords"]
			self.PARAMS['PREP_PARAMS']["TEST_RATIO"] = self.initial_params["TEST_RATIO"]
			self.PARAMS['PREP_PARAMS']["CV"] = self.initial_params["CV"]
			self.PARAMS['PREP_PARAMS']["SPLIT"] = self.initial_params["SPLIT"]
		except:
			pass

		# self.PARAMS['remove_stopwords'] = self.module.stopword
		# self.PARAMS['zemberek_stemmer'] =  False
		# self.PARAMS['zemberek_spellcheck'] = False
		return self.PARAMS


class CeleryService:
	def create_task(app):
		celery = Celery("train",
		                backend = app.config['CELERY_RESULT_BACKEND'],
		                broker = app.config['CELERY_BROKER_URL'])
		celery.conf.update(app.config)
		TaskBase = celery.Task
		
		class ContextTask(TaskBase):
			abstract = True
			
			def __call__(self, *args, **kwargs):
				with app.app_context():
					return TaskBase.__call__(self, *args, **kwargs)
		
		celery.Task = ContextTask
		return celery


class Root(dict):
	def __getattr__(self, key):
		try:
			return self[key]
		except KeyError as k:
			raise AttributeError(k)
	
	def __setattr__(self, key, value):
		self[key] = value
	
	def __delattr__(self, key):
		try:
			del self[key]
		except KeyError as k:
			raise AttributeError(k)
	
	def __repr__(self):
		return '<Root ' + dict.__repr__(self) + '>'