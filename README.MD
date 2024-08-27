# Python

### Install python 3.10

https://gist.github.com/rutcreate/c0041e842f858ceb455b748809763ddb

### Install environment

`python3 -m venv .venv`
`.venv/bin/pip install -r requirements.txt`

# Django

Generate base site : `django-admin startproject mysite`  
Generate sub site : `python manage.py startapp detections`  
Start site : `python manage.py runserver`

Update schema : `python manage.py makemigrations && python manage.py migrate`

Create first admin user : `python manage.py createsuperuser`  
Export
data :
`python manage.py dumpdata --format=yaml --exclude=admin --exclude=auth --exclude=sessions --exclude=contenttypes --exclude=detections`  
Import data : `python manage.py loaddata data.yaml`

# Dev

Start
kill old backoffice : `lsof -t -i tcp:8000 | xargs kill -9`
backoffice : `.venv/bin/python manage.py runserver_plus --cert-file ~/cert.pem --key-file ~/key.pem 8000`  
Kill processus : `pkill -9 ffmpeg`  
Start vision : `.venv/bin/python manage.py vision`

# Hailo

Convert pt to onnx : `.venv/bin/python scripts/quantize.py models/guinea-pig-chons-v12`
Convert onnx to hef :
`hailomz compile --ckpt guinea-pig-chons-v12.onnx --hw-arch hailo8l  --calib-path data --yaml hef_config.yaml --classes 4`

`wsl -e ./scripts/train.sh`
`cd /mnt/c/Users/Salon/Hailo/`
`./hailo_ai_sw_suite_docker_run.sh --resume`
`docker exec -i  hailo_ai_sw_suite_2024-07_container hailomz compile --ckpt ../shared_with_docker/followchon_back/models/guinea-pig-chons-v12.onnx --hw-arch hailo8l --calib-path ../shared_with_docker/followchon_back/datasets/chons-v12/train --yaml ../shared_with_docker/followchon_back/models/config/hef_config_yolov8n.yaml --classes 4 && mv /local/workspace/yolov8n.hef ../shared_with_docker/followchon_back/models/guinea-pig-chons-v12.hef`

# Server

Deployment statics files : `.venv/bin/python manage.py collectstatic`
`cd /home/jaden/Projects/followchon_back/`
`venv/bin/uwsgi --http :8080 --home /home/jaden/Projects/followchon_back/venv --chdir /home/jaden/Projects/followchon_back/ -w /home/jaden/Projects/followchon_back/admin.wsgi`