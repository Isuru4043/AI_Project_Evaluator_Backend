#!/bin/bash

python manage.py migrate
python manage.py collectstatic --noinput
gunicorn AI_Evaluator_Backend.wsgi --bind=0.0.0.0:$PORT
