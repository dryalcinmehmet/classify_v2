from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

from config import Config
from .service import CeleryService
from flask_request_id_header.middleware import RequestID


app = Flask(__name__)
app.config['REQUEST_ID_UNIQUE_VALUE_PREFIX'] = 'FOO-'
RequestID(app)
app.config.from_object(Config)
db = SQLAlchemy(app)

migrate = Migrate(app, db)
celery = CeleryService.create_task(app)

from app import routes, models


