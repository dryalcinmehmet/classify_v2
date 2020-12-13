#!/bin/bash

celery -A app.helpers worker -l info -f /app/logs/classify_v2_celery.log
