#!/bin/bash

git pull
pkill -9 ffmpeg
kill $(ps aux | grep 'vision_hailo' | awk '{print $2}')
venv/bin/python manage.py vision_hailo &