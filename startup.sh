#!/bin/bash

# Increase NGINX upload size limit for Azure App Service Python containers
if [ -d "/etc/nginx/conf.d" ]; then
    echo "client_max_body_size 50M;" > /etc/nginx/conf.d/upload_limit.conf
    service nginx reload || true
fi

python manage.py migrate
python manage.py collectstatic --noinput
gunicorn AI_Evaluator_Backend.wsgi --bind=0.0.0.0:$PORT
