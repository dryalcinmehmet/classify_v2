#!/bin/bash
source venv/bin/activate
set -e
flask db upgrade
/usr/local/bin/gunicorn app:app --workers 4 --bind :$APPLICATION_PORT --capture-output --preload
