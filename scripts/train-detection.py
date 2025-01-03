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

from scripts.combine import CombinedModel

load_dotenv()

# settings.update({'datasets_dir': ''})

runs_dir = 'runs'
models_dir = 'models'
metrics_dir = 'metrics'

train_previous_path = os.getenv('TRAIN_MODEL_PATH')
train_dataset_path = os.getenv('TRAIN_DATASET_PATH')
train_dataset_yaml_path = f"{train_dataset_path}/data.yaml"
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

end_node_names = (
    '/model.22/cv2.0/cv2.0.2/Conv',
    '/model.22/cv3.0/cv3.0.2/Conv',
    '/model.22/cv2.1/cv2.1.2/Conv',
    '/model.22/cv3.1/cv3.1.2/Conv',
    '/model.22/cv2.2/cv2.2.2/Conv',
    '/model.22/cv3.2/cv3.2.2/Conv'
)


def train(train_dataset_name, train_type, classes):
    train_filename = train_previous_path.replace('.pt', f'-{train_type}.pt') \
        if train_previous_path.startswith('models/') \
        else train_previous_path

    model = YOLO(train_filename)
    model.train(
        task="detect",
        data=train_dataset_yaml_path,
        epochs=50,
        imgsz=1024,
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
        classes=classes,
    )


def move_metric(metric_dir, train_name, train_type, metric_name, metric_ext):
    shutil.move(
        f'{metric_dir}/{metric_name}.{metric_ext}',
        f'{metrics_dir}/{metric_name}/{train_name}-{train_type}-{metric_name}.{metric_ext}'
    )


def move_best(model_train_best, model_pt):
    shutil.move(model_train_best, model_pt)

    print(f'Model yolo saved in {model_pt}')


def train_full(train_type, train_classes, model_pt):
    train_name = f"{train_dataset_base_name}-{train_type}"
    model_run_dir = f"{runs_dir}/{train_name}"
    model_train_best = f"{model_run_dir}/weights/best.pt"

    print(f"Start train '{train_type}' at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    train(train_name, train_type, train_classes)

    print(f"End train '{train_type}' at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    move_best(model_train_best, model_pt)

    move_metric(model_run_dir, train_dataset_base_name, train_type, 'confusion_matrix', 'png')
    move_metric(model_run_dir, train_dataset_base_name, train_type, 'confusion_matrix_normalized', 'png')
    move_metric(model_run_dir, train_dataset_base_name, train_type, 'F1_curve', 'png')
    move_metric(model_run_dir, train_dataset_base_name, train_type, 'P_curve', 'png')
    move_metric(model_run_dir, train_dataset_base_name, train_type, 'R_curve', 'png')
    move_metric(model_run_dir, train_dataset_base_name, train_type, 'PR_curve', 'png')
    move_metric(model_run_dir, train_dataset_base_name, train_type, 'results', 'csv')


def combine(models):
    return CombinedModel(
        [YOLO(model) for model in models]
    )


def export(model_pt, model_onnx):
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


def parse(model_onnx, model_har, class_count):
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

    command_copy_har = (
        "docker", "exec", "-i", f"hailo_ai_sw_suite_{hailo_sdk_version}_container",
        "mv", f"/local/workspace/yolov{model_base_version}n.har",
        f"/local/shared_with_docker/followchon_back/{model_har}"
    )

    for path in execute(command_parse):
        print(path, end="")

    for path in execute(command_copy_har):
        print(path, end="")

    print(f'Model har saved in {model_har}')


def build(model_har, model_hef, classes_count):
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

    command_copy_hef = (
        "docker", "exec", "-i", f"hailo_ai_sw_suite_{hailo_sdk_version}_container",
        "mv", f"/local/workspace/yolov{model_base_version}n.hef",
        f"/local/shared_with_docker/followchon_back/{model_hef}"
    )

    for path in execute(command_compile):
        print(path, end="")

    for path in execute(command_copy_hef):
        print(path, end="")

    print(f'Model hef saved in {model_hef}')


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


def purge(dir, pattern):
    for f in os.listdir(dir):
        if re.search(pattern, f):
            os.remove(os.path.join(dir, f))


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

    train_types = ['all', 'chons']
    train_classes = [[0], [1, 2]]

    for i, train_type in enumerate(train_types):
        if i == 0 and train_all or i == 1 and train_chons:

            train_name = f"{train_dataset_base_name}-{train_type}"
            model_pt = f"{models_dir}/{train_name}.pt"
            model_onnx = f"{models_dir}/{train_name}.onnx"
            model_har = f"{models_dir}/{train_name}.har"
            model_hef = f"{models_dir}/{train_name}.hef"
            file_stats = f"{metrics_dir}/duration/{train_name}.txt"

            if not os.path.exists(file_stats):
                with open(file_stats, "w") as file:
                    file.write("Train count : " + str(
                        len(list(Path(f"{train_dataset_path}/train/labels").glob("*.txt")))) + "\n")
                    file.write(
                        "Val count : " + str(len(list(Path(f"{train_dataset_path}/val/labels").glob("*.txt")))) + "\n")
                    file.write("Test count : " + str(
                        len(list(Path(f"{train_dataset_path}/test/labels").glob("*.txt")))) + "\n\n")

            if not os.path.exists(model_pt):
                with open(file_stats, "a") as file:
                    file.write("[Train] Start at : " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")

                train_full(train_type, train_classes[i], model_pt)

                with open(file_stats, "a") as file:
                    file.write("[Train] End at : " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")
                    file.write(f"[Train] {calc_time_h_m(train_start)}\n")

            export(model_pt, model_onnx)

            if os.getenv('TRAIN_STEP_PARSE') and not os.path.exists(model_har):
                parse(model_onnx, model_har, 5)

            if os.getenv('TRAIN_STEP_COMPILE') and not os.path.exists(model_hef):
                compile_start = datetime.now()
                with open(file_stats, "a") as file:
                    file.write("[Compile] Start at : " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")

                build(model_har, model_hef, 5)

                with open(file_stats, "a") as file:
                    file.write("[Compile] End at : " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")
                    file.write(f"[Compile] {calc_time_h_m(compile_start)}\n")

            if os.getenv('TRAIN_STEP_GIT'):
                if os.path.exists(model_hef):
                    commit(train_name, [model_pt, model_hef, f'{metrics_dir}/*'])

    if is_cached:
        try:
            purge(f'{train_dataset_path}/train', '*.npy')
            purge(f'{train_dataset_path}/val', '*.npy')
            purge(f'{train_dataset_path}/test', '*.npy')
        except Exception as ex:
            print(ex)

    print(f"End at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
