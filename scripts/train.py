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

train_previous_path = os.getenv('TRAIN_MODEL_PATH')
train_dataset_yaml_path = f"{os.getenv('TRAIN_DATASET_PATH')}/data.yaml"
train_dataset_name = os.getenv('TRAIN_DATASET_NAME')
train_device = os.getenv('TRAIN_DEVICE')

model_train_last = f"{runs_dir}/{os.getenv('TRAIN_DATASET_NAME')}/weights/last.pt"
model_pt = f"{models_dir}/{os.getenv('TRAIN_DATASET_NAME')}.pt"
model_onnx = f"{models_dir}/{os.getenv('TRAIN_DATASET_NAME')}.onnx"
model_hef = f"{models_dir}/{os.getenv('TRAIN_DATASET_NAME')}.hef"


def train():
    model = YOLO(train_previous_path)
    model.train(
        data=train_dataset_yaml_path,
        epochs=50,
        imgsz=1024,
        name=train_dataset_name,
        verbose=True,
        save=False,
        project=runs_dir,
        exist_ok=True,
        device=train_device,
        workers=8
    )

    shutil.move(model_train_last, model_pt)

    print(f'Model yolo saved in {model_pt}')


def export():
    model = YOLO(model_pt)
    model.export(format="onnx")

    print(f'Model onnx saved in {model_onnx}')


def execute(cmd):
    print(f'Execute : {cmd}')

    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def build():
    command_build = (
        "docker", "exec", "-i", "hailo_ai_sw_suite_2024-07_container",
        "hailomz", "compile",
        "--ckpt", f"../shared_with_docker/followchon_back/{model_onnx}",
        "--hw-arch", "hailo8l",
        "--calib-path", f"../shared_with_docker/followchon_back/{os.getenv('TRAIN_DATASET_PATH')}/train",
        "--yaml", "../shared_with_docker/followchon_back/models/config/hef_config_yolov8n.yaml",
        "--classes", "4",
    )

    command_copy = (
        "docker", "exec", "-i", "hailo_ai_sw_suite_2024-07_container",
        "mv", "/local/workspace/yolov8n.hef", f"../shared_with_docker/followchon_back/{model_hef}"
    )

    for path in execute(command_build):
        print(path, end="")

    for path in execute(command_copy):
        print(path, end="")

    print(f'Model hef saved in {model_hef}')


def commit():
    for path in execute(('git', 'pull')):
        print(path, end="")

    for path in execute(('git', 'add', model_pt, model_onnx, model_hef)):
        print(path, end="")

    for path in execute(('git', 'commit', '-m', os.getenv('TRAIN_DATASET_NAME'))):
        print(path, end="")

    for path in execute(('git', 'push')):
        print(path, end="")


if __name__ == '__main__':
    if torch.cuda.is_available():
        print(":D GPU is available")
    else:
        print("T_T GPU is not available")

    print(f"Start at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # train()
    # export()
    build()
    commit()

    print(f"End at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
