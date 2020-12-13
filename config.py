import os
basedir = os.path.abspath(os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'NO-ACCESS'
    SQLALCHEMY_DATABASE_URI = 'postgresql+psycopg2://{user}:{pw}@{url}/{db}'.format(
                                                    user=os.environ.get('DATABASE_USER'),
                                                    pw=os.environ.get('DATABASE_PASSWORD'),
                                                    url=os.environ.get('DATABASE_HOST'),
                                                    db=os.environ.get('DATABASE_NAME'))
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_POOL_TIMEOUT = 100
    
    CELERY_BROKER_URL =  'redis://{user}:{pw}@{host}:{port}/{db}'.format(
                                                              user = os.environ.get("REDIS_PASSWORD"),
                                                              pw = os.environ.get("REDIS_USER"),
                                                              host = os.environ.get("REDIS_HOST"),
                                                              port = os.environ.get("REDIS_PORT"),
                                                              db = os.environ.get("REDIS_DB")
                                                              )
                                     

    CELERY_RESULT_BACKEND = 'redis://{user}:{pw}@{host}:{port}/{db}'.format(
                                                              user = os.environ.get("REDIS_PASSWORD"),
                                                              pw = os.environ.get("REDIS_USER"),
                                                              host = os.environ.get("REDIS_HOST"),
                                                              port = os.environ.get("REDIS_PORT"),
                                                              db = os.environ.get("REDIS_DB")
                                                              )