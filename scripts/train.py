import os
import shutil
import subprocess
from datetime import datetime

import onnx
import torch
from dotenv import load_dotenv
from onnx import __version__, IR_VERSION
from onnx.defs import onnx_opset_version
from ultralytics import YOLO

load_dotenv()

# settings.update({'datasets_dir': ''})

runs_dir = 'runs'
models_dir = 'models'

train_previous_path = os.getenv('TRAIN_MODEL_PATH')
train_dataset_yaml_path = f"{os.getenv('TRAIN_DATASET_PATH')}/data.yaml"
train_dataset_name = os.getenv('TRAIN_DATASET_NAME')
train_device = os.getenv('TRAIN_DEVICE')
train_step_train = os.getenv('TRAIN_STEP_TRAIN')
train_step_export = os.getenv('TRAIN_STEP_EXPORT')
train_step_build = os.getenv('TRAIN_STEP_BUILD')
train_step_git = os.getenv('TRAIN_STEP_GIT')
train_resume = os.getenv('TRAIN_RESUME')
hailo_sdk_version = os.getenv('TRAIN_SDK_VERSION')

model_train_last = f"{runs_dir}/{os.getenv('TRAIN_DATASET_NAME')}/weights/best.pt"
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
        save=True,
        resume=train_resume,
        project=runs_dir,
        exist_ok=True,
        device=train_device,
        workers=8
    )

    shutil.move(model_train_last, model_pt)

    print(f'Model yolo saved in {model_pt}')


def export():
    model = YOLO(model_pt)
    model.export(format="onnx", opset=20)

    original_model = onnx.load(model_onnx)
    print(f"From : ONNX ir_version={original_model.ir_version}")

    # cnverted_model = version_converter.convert_version(original_model, 9)
    # print(f"To : ONNX version={converted_model.model_version}, ir_version={converted_model.ir_version}")
    # onnx.save_model(converted_model, model_onnx)

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
        "docker", "exec", "-i", f"hailo_ai_sw_suite_{hailo_sdk_version}_container",
        "hailomz", "compile",
        "--ckpt", f"../shared_with_docker/followchon_back/{model_onnx}",
        "--hw-arch", "hailo8l",
        "--calib-path", f"../shared_with_docker/followchon_back/{os.getenv('TRAIN_DATASET_PATH')}/val",
        "--yaml",
        f"../shared_with_docker/followchon_back/models/config/{os.getenv('TRAIN_MODEL_BASE')}/hef_config_n.yaml",
        "--classes", os.getenv('TRAIN_CLASSES'),
    )

    command_copy = (
        "docker", "exec", "-i", "hailo_ai_sw_suite_2024-07_container",
        "mv", f"/local/workspace/{os.getenv('TRAIN_MODEL_BASE')}n.hef",
        f"../shared_with_docker/followchon_back/{model_hef}"
    )

    for path in execute(command_build):
        print(path, end="")

    for path in execute(command_copy):
        print(path, end="")

    print(f'Model hef saved in {model_hef}')


def commit():
    for path in execute(('git', 'pull')):
        print(path, end="")

    for path in execute(('git', 'add', model_pt, model_hef)):
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

    print("PyTorch version :", torch.__version__)
    print("CUDA Devices :", torch.cuda.device_count())
    print(f"ONNX version={__version__!r}, opset={onnx_opset_version()}, ir_version={IR_VERSION}")

    print(f"Start at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if train_step_train:
        train()

    if train_step_export:
        export()

    if train_step_build:
        build()

    if train_step_git:
        commit()

    print(f"End at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
