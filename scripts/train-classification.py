import gc
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import torch
from dotenv import load_dotenv
from onnx import __version__, IR_VERSION
from onnx.defs import onnx_opset_version
from ultralytics import YOLO

load_dotenv()

# settings.update({'datasets_dir': ''})

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_VISIBLE_DEVICE"] = "0"

runs_dir = 'runs'
models_dir = 'models'
metrics_dir = 'metrics'

train_previous_path = os.getenv('TRAIN_MODEL_PATH')
train_dataset_path = os.getenv('TRAIN_DATASET_PATH')
train_dataset_base_name = os.getenv('TRAIN_DATASET_NAME')
train_device = os.getenv('TRAIN_DEVICE')
train_resume = os.getenv('TRAIN_RESUME')
train_all = os.getenv('TRAIN_ALL')
train_chons = os.getenv('TRAIN_CHONS')
hailo_sdk_version = os.getenv('TRAIN_SDK_VERSION')
model_base_version = os.getenv('TRAIN_MODEL_BASE_VERSION')
model_nms_version = os.getenv('TRAIN_MODEL_NMS_VERSION')
train_calib_dir = os.getenv('TRAIN_CALIB_DIR')

is_cached = False

gc.collect()
torch.cuda.empty_cache()


def train(train_dataset_name):
    model = YOLO(train_previous_path)
    model.train(
        task='classify',
        data=train_dataset_path,
        epochs=50,
        batch=12,
        imgsz=480,
        name=train_dataset_name,
        verbose=True,
        save=True,
        cache='disk' if is_cached else None,
        plots=True,
        resume=train_resume,
        project=runs_dir,
        exist_ok=True,
        device=train_device,
        workers=8,
    )


def move_metric(metric_dir, train_name, metric_name, metric_ext):
    shutil.move(
        f'{metric_dir}/{metric_name}.{metric_ext}',
        f'{metrics_dir}/{metric_name}/{train_name}-{metric_name}.{metric_ext}'
    )


def move_best(model_train_best, model_pt):
    shutil.move(model_train_best, model_pt)

    print(f'Model yolo saved in {model_pt}')


def train_full(model_pt):
    train_name = f"{train_dataset_base_name}"
    model_run_dir = f"{runs_dir}/{train_name}"
    model_train_best = f"{model_run_dir}/weights/best.pt"

    print(f"Start train at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    train(train_name)

    print(f"End train at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    move_best(model_train_best, model_pt)

    move_metric(model_run_dir, train_dataset_base_name, 'confusion_matrix', 'png')
    move_metric(model_run_dir, train_dataset_base_name, 'confusion_matrix_normalized', 'png')
    move_metric(model_run_dir, train_dataset_base_name, 'results', 'csv')


def execute(cmd):
    print(f'Execute : {cmd}')

    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def commit(train_dataset_name, files):
    for file in files:
        for path in execute(('git', 'add', file)):
            print(path, end="")

    try:
        for path in execute(('git', 'commit', '-m', train_dataset_name)):
            print(path, end="")
    except Exception as ex:
        print(ex)

    try:
        for path in execute(('git', 'fetch', 'origin')):
            print(path, end="")
    except Exception as ex:
        print(ex)

    try:
        for path in execute(('git', 'pull')):
            print(path, end="")
    except Exception as ex:
        print(ex)

    try:
        for path in execute(('git', 'push')):
            print(path, end="")
    except Exception as ex:
        print(ex)


def calc_time_h_m(dt):
    diff = datetime.now() - dt
    hours, seconds = divmod(diff.total_seconds(), 3600)
    minutes = seconds // 60
    return f"{int(hours)} hours and {int(minutes)} minutes"


if __name__ == '__main__':
    if torch.cuda.is_available():
        print(":D GPU is available")
    else:
        print("T_T GPU is not available")

    print("PyTorch version :", torch.__version__)
    print("CUDA Devices :", torch.cuda.device_count())
    print(f"ONNX version={__version__!r}, opset={onnx_opset_version()}, ir_version={IR_VERSION}")
    print(f"Start at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    train_start = datetime.now()
    train_end = None
    compile_start = None
    compile_end = None
    training = False

    train_classes = [1, 2]

    train_name = f"{train_dataset_base_name}"
    model_pt = f"{models_dir}/{train_name}.pt"
    file_stats = f"{metrics_dir}/duration/{train_name}.txt"

    if not os.path.exists(file_stats):
        with open(file_stats, "w") as file:
            file.write("Train noisette count : " + str(
                len(list(Path(f"{train_dataset_path}/train/noisette").glob("*.*")))) + "\n")
            file.write("Train stitch count : " + str(
                len(list(Path(f"{train_dataset_path}/train/stitch").glob("*.*")))) + "\n")
            file.write(
                "Val noisette count : " + str(len(list(Path(f"{train_dataset_path}/val/noisette").glob("*.*")))) + "\n")
            file.write(
                "Val stitch count : " + str(len(list(Path(f"{train_dataset_path}/val/stitch").glob("*.*")))) + "\n")
            file.write("Test noisette count : " + str(
                len(list(Path(f"{train_dataset_path}/test/noisette").glob("*.*")))) + "\n")
            file.write("Test stitch count : " + str(
                len(list(Path(f"{train_dataset_path}/test/stitch").glob("*.*")))) + "\n\n")

    try:
        if not os.path.exists(model_pt):
            with open(file_stats, "a") as file:
                file.write("[Train] Start at : " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")

            training = True
            train_full(model_pt)

            with open(file_stats, "a") as file:
                file.write("[Train] End at : " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")
                file.write(f"[Train] {calc_time_h_m(train_start)}\n")

        if os.getenv('TRAIN_STEP_GIT'):
            if os.path.exists(model_pt):
                commit(train_name, [model_pt, f'{metrics_dir}/*'])

    except Exception as ex:
        print(ex)
        with open(file_stats, "a") as file:
            file.write("[Train] Fail at : " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")
            file.write(f"[Train] {calc_time_h_m(train_start)}\n")

    print(f"End at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if os.getenv('TRAIN_SHUTDOWN') and training:
        print(f"Shutdown")
        os.system("shutdown /s /t 600")
