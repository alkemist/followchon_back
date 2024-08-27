#!/bin/bash

# cd /mnt/c/Users/Salon/Hailo/
docker exec -i  hailo_ai_sw_suite_2024-07_container hailomz compile --ckpt ../shared_with_docker/followchon_back/models/guinea-pig-chons-v12.onnx --hw-arch hailo8l --calib-path ../shared_with_docker/followchon_back/datasets/chons-v12/train --yaml ../shared_with_docker/followchon_back/models/config/hef_config_yolov8n.yaml --classes 4
