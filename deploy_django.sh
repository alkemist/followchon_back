#!/bin/bash

git pull
venv/bin/python manage.py migrate
sudo systemctl restart uwsgi