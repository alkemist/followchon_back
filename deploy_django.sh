#!/bin/bash

git pull
cp sqlite3.db sqlite3_backup.db
venv/bin/python manage.py migrate
sudo systemctl restart uwsgi