# Python

### Install python 3.10
https://gist.github.com/rutcreate/c0041e842f858ceb455b748809763ddb

### Install environment
`python3 -m venv .venv`
`.venv/bin/python -m pip install -r requirements.txt`

# Django

Generate base site : `django-admin startproject mysite`  
Generate sub site : `python manage.py startapp detections`  
Start site : `python manage.py runserver`

Update schema : `python manage.py makemigrations && python manage.py migrate`

Create first admin user : `python manage.py createsuperuser`  