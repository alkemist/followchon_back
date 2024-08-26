import os
import shutil
import subprocess
from datetime import datetime

import torch
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

runs_dir = 'runs'
models_dir = 'models'


def train():
    model = YOLO(os.getenv('TRAIN_MODEL_PATH'))
    model.train(
        data=os.getenv('TRAIN_DATASET_PATH'),
        epochs=50,
        imgsz=1024,
        name=os.getenv('TRAIN_DATASET_NAME'),
        verbose=True,
        save=False,
        project=runs_dir,
        exist_ok=True,
        device=os.getenv('TRAIN_DEVICE'),
        workers=8
    )


def end():
    model_train_last = f"{runs_dir}/{os.getenv('TRAIN_DATASET_NAME')}/weights/last.pt"
    model_pt = f"{models_dir}/{os.getenv('TRAIN_DATASET_NAME')}.pt"
    model_onnx = f"{models_dir}/{os.getenv('TRAIN_DATASET_NAME')}.onnx"
    model_hef = f"{models_dir}/{os.getenv('TRAIN_DATASET_NAME')}.hef"

    shutil.move(model_train_last, model_pt)
    print(f'Model yolo saved in {model_pt}')

    model = YOLO(model_pt)
    model.export(format="onnx")

    command = (
        f"docker exec -i  hailo_ai_sw_suite_2024-07_container hailomz compile "
        f" --ckpt ../shared_with_docker/followchon_back/{model_onnx} "
        f"--hw-arch hailo8l "
        f"--calib-path ../shared_with_docker/followchon_back/{os.getenv('TRAIN_DATASET_PATH')}/train "
        f"--yaml ../shared_with_docker/followchon_back/models/config/hef_config_yolov8n.yaml "
        f"--classes 4 "
        f"&& mv /local/workspace/yolov8n.hef ../shared_with_docker/followchon_back/{model_hef}"
    )

    subprocess.Popen(command.split(" "),
                     stdout=subprocess.PIPE,
                     universal_newlines=True)

    print(f'Model onnx saved in {model_onnx}')
    print(f'Model hef saved in {model_hef}')


if __name__ == '__main__':
    if torch.cuda.is_available():
        print(":D GPU is available")
    else:
        print("T_T GPU is not available")

    print(f"Start at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        train()
    except PermissionError:
        print('Permission error')

    end()

    print(f"End at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
