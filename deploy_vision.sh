#!/bin/bash

pkill -9 ffmpeg
kill $(ps aux | grep 'vision_hailo' | awk '{print $2}')
cd /home/jaden/Projects/followchon_back
venv/bin/python manage.py vision_hailo &
