import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import torch
from dotenv import load_dotenv
from onnx import __version__, IR_VERSION
from onnx.defs import onnx_opset_version
from ultralytics import YOLO

# settings.update({'datasets_dir': ''})

load_dotenv()

train_previous_path = os.getenv('TRAIN_DETECT_MODEL_PATH')
train_dataset_path = os.getenv('TRAIN_DETECT_DATASET_PATH')
train_dataset_yaml_path = f"{train_dataset_path}/data.yaml"
train_name = os.getenv('TRAIN_DETECT_DATASET_NAME')
train_device = os.getenv('TRAIN_DEVICE')
train_resume = os.getenv('TRAIN_RESUME')
train_all = os.getenv('TRAIN_ALL')
train_chons = os.getenv('TRAIN_CHONS')
hailo_sdk_version = os.getenv('TRAIN_SDK_VERSION')
model_base_version = os.getenv('TRAIN_MODEL_BASE_VERSION')
model_nms_version = os.getenv('TRAIN_MODEL_NMS_VERSION')
train_calib_dir = os.getenv('TRAIN_CALIB_DIR')

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_VISIBLE_DEVICE"] = "0"

runs_dir = 'runs'
models_dir = 'models'
metrics_dir = 'metrics'

is_cached = True
is_ok = False

model_pt = f"{models_dir}/{train_name}.pt"
model_onnx = f"{models_dir}/{train_name}.onnx"
model_har = f"{models_dir}/{train_name}.har"
model_hef = f"{models_dir}/{train_name}.hef"
file_stats = f"{metrics_dir}/duration/{train_name}.txt"

model_run_dir = f"{runs_dir}/{train_name}"
model_train_best = f"{model_run_dir}/weights/best.pt"


def log_version():
    if torch.cuda.is_available():
        print(":D GPU is available")
    else:
        print("T_T GPU is not available")

    print("PyTorch version :", torch.__version__)
    print("CUDA Devices :", torch.cuda.device_count())
    print(f"ONNX version={__version__!r}, opset={onnx_opset_version()}, ir_version={IR_VERSION}")


def tune():
    tune_start = log_start('Tune')

    model = YOLO(train_previous_path)

    best_result = model.tune(
        data={
            'data': train_dataset_yaml_path,
            'epochs': 10,  # Nombre d'époques pour chaque essai
            'imgsz': 1024,  # Taille de l'image
            'save': False,  # Ne pas enregistrer les modèles pendant le tuning
            'val': False,  # Ne pas faire de validation pendant le tuning
        },
        use_ray=True
    )
    log_end('Tune', tune_start)

    return YOLO(best_result.best['model'])


def train(model, task, imgsz, classes=None):
    if os.getenv('TRAIN_STEP_TRAIN') and not os.path.exists(model_pt):
        train_start = log_start('Train')

        model.train(
            task=task,
            data=train_dataset_path,
            epochs=50,
            imgsz=imgsz,
            name=train_name,
            verbose=True,
            save=True,
            cache='disk' if is_cached else None,
            plots=True,
            resume=train_resume,
            project=runs_dir,
            exist_ok=True,
            device=train_device,
            workers=8,
            classes=classes,
        )
        log_end('Train', train_start)

        shutil.move(model_train_best, model_pt)
        print(f'-- Model yolo saved in {model_pt}')

        is_ok = True


def move_metrics(metrics_configs):
    for [metric_name, metric_ext] in metrics_configs:
        shutil.move(
            f'{model_run_dir}/{metric_name}.{metric_ext}',
            f'{metrics_dir}/{metric_name}/{train_name}-{metric_name}.{metric_ext}'
        )


def log_files_counts(dirs_first_level, dirs_second_level, filter):
    if not os.path.exists(file_stats):
        with open(file_stats, "w") as file:
            for second_dir in dirs_second_level:
                for first_dir in dirs_first_level:
                    title = second_dir.title() if len(dirs_second_level) > 1 else ''

                    file.write(
                        f"{first_dir.title()} : {title} count"
                        + str(len(list(Path(f"{train_dataset_path}/{first_dir}/{second_dir}").glob(filter))))
                        + "\n"
                    )

                file.write("\n")
            file.write("\n")


def purge_files(dir, pattern):
    for f in os.listdir(dir):
        if re.search(pattern, f):
            try:
                os.remove(os.path.join(dir, f))
            except Exception as ex:
                print(ex)


def purge_cache(dirs_first_level, dirs_second_level):
    if is_cached:
        for second_dir in dirs_second_level:
            for first_dir in dirs_first_level:
                purge_files(f'{train_dataset_path}/{first_dir}/{second_dir}', '*.npy')


def calc_time_h_m(dt):
    diff = datetime.now() - dt
    hours, seconds = divmod(diff.total_seconds(), 3600)
    minutes = seconds // 60
    return f"{int(hours)} hours and {int(minutes)} minutes"


def log_start(type):
    start = datetime.now()
    with open(file_stats, "a") as file:
        file.write(f"[{type}] Start at : " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")

    return start


def log_end(type, date_start):
    with open(file_stats, "a") as file:
        file.write(f"[{type}] End at : " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")
        file.write(f"[{type}] {calc_time_h_m(date_start)}\n")


def execute(cmd):
    print(f'Execute : {cmd}')

    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def end():
    if os.getenv('TRAIN_SHUTDOWN') and is_ok:
        print(f"Shutdown")
        os.system("shutdown /s /t 600")


def commit_files(files):
    if os.getenv('TRAIN_STEP_GIT'):
        for file in files:
            for path in execute(('git', 'add', file)):
                print(path, end="")

        try:
            for path in execute(('git', 'commit', '-m', train_name)):
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


def export():
    model = YOLO(model_pt)
    model.export(format="onnx")

    print(f'Model onnx saved in {model_onnx}')


def parse(class_count):
    if os.getenv('TRAIN_STEP_PARSE') and not os.path.exists(model_har):
        command_parse = (
            "docker", "exec", "-i", f"hailo_ai_sw_suite_{hailo_sdk_version}_container",
            "hailomz", "parse",
            "--hw-arch", "hailo8l",
            "--ckpt", f"/local/shared_with_docker/followchon_back/{model_onnx}",
            # "--start-node-names", "images",
            # "--end-node-names") + end_node_names + (
            "--yaml",
            f"/local/shared_with_docker/followchon_back/models/config/{model_base_version}/hef_config_n_{model_nms_version}-{class_count}.yaml",
            # "/local/workspace/hailo_model_zoo/hailo_model_zoo/cfg/networks/yolov8n.yaml",
            # f"yolov{model_base_version}n",
        )

        parse_start = log_start('Parse')

        for path in execute(command_parse):
            print(path, end="")

        log_end('Parse', parse_start)

        command_copy_har = (
            "docker", "exec", "-i", f"hailo_ai_sw_suite_{hailo_sdk_version}_container",
            "mv", f"/local/workspace/yolov{model_base_version}n.har",
            f"/local/shared_with_docker/followchon_back/{model_har}"
        )

        for path in execute(command_copy_har):
            print(path, end="")

        print(f'Model har saved in {model_har}')


def build(classes_count):
    if os.getenv('TRAIN_STEP_COMPILE') and not os.path.exists(model_hef):
        command_compile = (
            "docker", "exec", "-i", f"hailo_ai_sw_suite_{hailo_sdk_version}_container",
            "hailomz", "compile",
            "--hw-arch", "hailo8l",
            "--har", f"/local/shared_with_docker/followchon_back/{model_har}",
            # "--ckpt", f"/local/shared_with_docker/followchon_back/{model_onnx}",
            "--classes", str(classes_count),
            # "--start-node-names", "images",
            # "--end-node-names") + end_node_names + (
            "--calib-path", f"/local/shared_with_docker/followchon_back/{train_dataset_path}/{train_calib_dir}",
            # "--model-script",
            # f"/local/shared_with_docker/followchon_back/models/config/{model_base_version}/yolo.alls",
            "--yaml",
            f"/local/shared_with_docker/followchon_back/models/config/{model_base_version}/hef_config_n_{model_nms_version}-{classes_count}.yaml",
            # "/local/workspace/hailo_model_zoo/hailo_model_zoo/cfg/networks/yolov8n.yaml",
            "--performance",
            # f"yolov{model_base_version}n",
        )

        compile_start = log_start('Compile')

        for path in execute(command_compile):
            print(path, end="")

        log_end('Compile', compile_start)

        command_copy_hef = (
            "docker", "exec", "-i", f"hailo_ai_sw_suite_{hailo_sdk_version}_container",
            "mv", f"/local/workspace/yolov{model_base_version}n.hef",
            f"/local/shared_with_docker/followchon_back/{model_hef}"
        )

        for path in execute(command_copy_hef):
            print(path, end="")

        print(f'Model hef saved in {model_hef}')

        is_ok = True
