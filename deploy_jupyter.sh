#!/bin/bash

kill-port 8888
cd /home/jaden/Projects/followchon_back
venv/bin/python manage.py shell_plus --lab &