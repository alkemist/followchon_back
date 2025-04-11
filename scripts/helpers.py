import glob
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import torch
from dotenv import load_dotenv
from onnx import __version__, IR_VERSION
from onnx.defs import onnx_opset_version
from ultralytics import YOLO, settings

settings.update({'datasets_dir': ''})

load_dotenv()

train_device = os.getenv('TRAIN_DEVICE')
train_resume = os.getenv('TRAIN_RESUME', False)
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


def log_version():
    if torch.cuda.is_available():
        print(":D GPU is available")
    else:
        print("T_T GPU is not available")

    print("PyTorch version :", torch.__version__)
    print("CUDA Devices :", torch.cuda.device_count())
    print(f"ONNX version={__version__!r}, opset={onnx_opset_version()}, ir_version={IR_VERSION}")


def train(
        task,
        imgsz,
        train_previous_path,
        train_dataset_data_path,
        train_name,
        metrics_configs,
        classes=None
):
    model_pt = f"{models_dir}/{train_name}.pt"
    model_run_dir = f"{runs_dir}/{train_name}"
    model_train_best = f"{model_run_dir}/weights/best.pt"
    best_hyperparameters_path = f'{models_dir}/{task}_best_hyperparameters.yaml'

    if os.getenv('TRAIN_STEP_TRAIN') and not os.path.exists(model_pt):
        data_path = os.path.abspath(train_dataset_data_path)

        train_params = {
            'data': data_path,
            'task': task,
            'imgsz': imgsz,
            'verbose': True,
            'cache': 'disk' if is_cached else None,
            'project': runs_dir,
            'exist_ok': True,
            'device': train_device,
            'workers': 8,
            'classes': classes,
        }

        model = YOLO(train_previous_path)

        if os.getenv('TRAIN_STEP_TUNE'):
            tune_train_name = f"{train_name}-tune"

            tune_start = log_start(train_name, 'Tune')
            model.tune(
                **train_params,
                name=tune_train_name,
                epochs=10,
                iterations=5,
                use_ray=False,
                val=False,
                plots=False,
                save=False,
            )
            log_end(train_name, 'Tune', tune_start)

            shutil.move(
                f'{runs_dir}/{tune_train_name}/best_hyperparameters.yaml',
                best_hyperparameters_path
            )

        train_start = log_start(train_name, 'Train')
        model.train(
            **train_params,
            name=train_name,
            epochs=50,
            resume=train_resume,
            plots=True,
            save=True,
            cfg=best_hyperparameters_path if os.path.exists(best_hyperparameters_path) else None,
        )
        log_end(train_name, 'Train', train_start)

        shutil.move(model_train_best, model_pt)
        print(f'-- Model yolo saved in {model_pt}')

        for [metric_name, metric_ext] in metrics_configs:
            shutil.move(
                f'{model_run_dir}/{metric_name}.{metric_ext}',
                f'{metrics_dir}/{metric_name}/{train_name}-{metric_name}.{metric_ext}'
            )

        return True


def log_files_counts(train_name, train_dataset_path, dirs_first_level, dirs_second_level, filter):
    file_stats = f"{metrics_dir}/duration/{train_name}.txt"

    if not os.path.exists(file_stats):
        with open(file_stats, "w") as file:
            for second_dir in dirs_second_level:
                for first_dir in dirs_first_level:
                    title = second_dir.title() if len(dirs_second_level) > 1 else ''

                    file.write(
                        f"{first_dir.title()} {title} count : "
                        + str(len(list(Path(f"{train_dataset_path}/{first_dir}/{second_dir}").glob(filter))))
                        + "\n"
                    )

                file.write("\n")
            file.write("\n")


def purge_files(dir, pattern):
    print(f'-- Clean {dir}')
    try:
        for file in glob.glob(os.path.join(dir, pattern)):
            os.remove(file)
    except Exception as ex:
        print(ex)


def purge_cache(train_dataset_path, dirs_first_level, dirs_second_level):
    if is_cached:
        for second_dir in dirs_second_level:
            for first_dir in dirs_first_level:
                purge_files(f'{train_dataset_path}/{first_dir}/{second_dir}', '*.npy')


def calc_time_h_m(dt):
    diff = datetime.now() - dt
    hours, seconds = divmod(diff.total_seconds(), 3600)
    minutes = seconds // 60
    return f"{int(hours)} hours and {int(minutes)} minutes"


def log_start(train_name, type):
    file_stats = f"{metrics_dir}/duration/{train_name}.txt"

    start = datetime.now()
    print(f"-- Start {type} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    with open(file_stats, "a") as file:
        file.write(f"[{type}] Start at : " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")

    return start


def log_end(train_name, type, date_start):
    file_stats = f"{metrics_dir}/duration/{train_name}.txt"
    print(f"-- End {type} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    with open(file_stats, "a") as file:
        file.write(f"[{type}] End at : " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")
        file.write(f"[{type}] {calc_time_h_m(date_start)}\n")


def execute(cmd):
    print(f'-- Execute : {cmd}')

    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def end(is_ok):
    if os.getenv('TRAIN_SHUTDOWN') and is_ok:
        print(f"-- Shutdown")
        os.system("shutdown /s /t 600")


def commit_files(train_name, files):
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


def export(train_name):
    model_pt = f"{models_dir}/{train_name}.pt"
    model_onnx = f"{models_dir}/{train_name}.onnx"

    model = YOLO(model_pt)
    model.export(format="onnx")

    print(f'-- Model onnx saved in {model_onnx}')


def parse(train_name, class_count):
    model_onnx = f"{models_dir}/{train_name}.onnx"
    model_har = f"{models_dir}/{train_name}.har"

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

        parse_start = log_start(train_name, 'Parse')

        for path in execute(command_parse):
            print(path, end="")

        log_end(train_name, 'Parse', parse_start)

        command_copy_har = (
            "docker", "exec", "-i", f"hailo_ai_sw_suite_{hailo_sdk_version}_container",
            "mv", f"/local/workspace/yolov{model_base_version}n.har",
            f"/local/shared_with_docker/followchon_back/{model_har}"
        )

        for path in execute(command_copy_har):
            print(path, end="")

        print(f'-- Model har saved in {model_har}')


def build(train_name, train_dataset_path, classes_count):
    model_har = f"{models_dir}/{train_name}.har"
    model_hef = f"{models_dir}/{train_name}.hef"

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

        compile_start = log_start(train_name, 'Compile')

        for path in execute(command_compile):
            print(path, end="")

        log_end(train_name, 'Compile', compile_start)

        command_copy_hef = (
            "docker", "exec", "-i", f"hailo_ai_sw_suite_{hailo_sdk_version}_container",
            "mv", f"/local/workspace/yolov{model_base_version}n.hef",
            f"/local/shared_with_docker/followchon_back/{model_hef}"
        )

        for path in execute(command_copy_hef):
            print(path, end="")

        print(f'-- Model hef saved in {model_hef}')

        return True
