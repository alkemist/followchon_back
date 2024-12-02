import os
import re
import shutil
import subprocess
from datetime import datetime

import torch
from dotenv import load_dotenv
from onnx import __version__, IR_VERSION
from onnx.defs import onnx_opset_version
from ultralytics import YOLO

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
hailo_sdk_version = os.getenv('TRAIN_SDK_VERSION')
model_base_version = os.getenv('TRAIN_MODEL_BASE_VERSION')
model_nms_version = os.getenv('TRAIN_MODEL_NMS_VERSION')
train_calib_dir = os.getenv('TRAIN_CALIB_DIR')

end_node_names = (
    '/model.22/cv2.0/cv2.0.2/Conv',
    '/model.22/cv3.0/cv3.0.2/Conv',
    '/model.22/cv2.1/cv2.1.2/Conv',
    '/model.22/cv3.1/cv3.1.2/Conv',
    '/model.22/cv2.2/cv2.2.2/Conv',
    '/model.22/cv3.2/cv3.2.2/Conv'
)

trains_classes = [
    {'name': 'all', 'classes': [0, 3, 4], 'class_count': 5},
    {'name': 'chons', 'classes': [1, 2], 'class_count': 5},
]


def train(train_dataset_name, train_type, classes):
    train_filename = train_previous_path.replace('.pt', f'-{train_type}.pt') \
        if train_previous_path.startswith('models/') \
        else train_previous_path

    model = YOLO(train_filename)
    model.train(
        data=train_dataset_yaml_path,
        epochs=50,
        imgsz=1024,
        name=train_dataset_name,
        verbose=True,
        save=True,
        cache='disk',
        plots=True,
        resume=train_resume,
        project=runs_dir,
        exist_ok=True,
        device=train_device,
        workers=8,
        classes=classes,
    )


def move_metric(metric_dir, train_name, train_type, metric_file):
    shutil.move(
        f'{metric_dir}/{metric_file}',
        f'{metrics_dir}/{train_type}/{train_name}-{metric_file}'
    )


def move(model_train_best, model_pt):
    shutil.move(model_train_best, model_pt)

    print(f'Model yolo saved in {model_pt}')


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


def commit(train_dataset_name, model_pt, model_hef):
    for path in execute(('git', 'pull')):
        print(path, end="")

    for path in execute(('git', 'add', model_pt, model_hef, f'{metrics_dir}/*')):
        print(path, end="")

    for path in execute(('git', 'commit', '-m', train_dataset_name)):
        print(path, end="")

    for path in execute(('git', 'push')):
        print(path, end="")


def purge(dir, pattern):
    for f in os.listdir(dir):
        if re.search(pattern, f):
            os.remove(os.path.join(dir, f))


if __name__ == '__main__':
    if torch.cuda.is_available():
        print(":D GPU is available")
    else:
        print("T_T GPU is not available")

    print("PyTorch version :", torch.__version__)
    print("CUDA Devices :", torch.cuda.device_count())
    print(f"ONNX version={__version__!r}, opset={onnx_opset_version()}, ir_version={IR_VERSION}")

    print(f"Start at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    for train_classes in trains_classes:
        train_name = f"{train_dataset_base_name}-{train_classes['name']}"
        model_run_dir = f"{runs_dir}/{train_name}"
        model_train_best = f"{model_run_dir}/weights/best.pt"
        model_pt = f"{models_dir}/{train_name}.pt"
        model_onnx = f"{models_dir}/{train_name}.onnx"

        if os.getenv('TRAIN_STEP_TRAIN'):
            if not os.path.exists(model_pt):
                train(train_name, train_classes['name'], train_classes['classes'])
                move(model_train_best, model_pt)

                move_metric(model_run_dir, train_dataset_base_name, train_classes['name'], 'confusion_matrix.png')
                move_metric(model_run_dir, train_dataset_base_name, train_classes['name'],
                            'confusion_matrix_normalized.png')
                move_metric(model_run_dir, train_dataset_base_name, train_classes['name'], 'results.csv')
                move_metric(model_run_dir, train_dataset_base_name, train_classes['name'], 'results.png')

        if os.getenv('TRAIN_STEP_EXPORT'):
            export(model_pt, model_onnx)

    for train_classes in trains_classes:
        train_name = f"{train_dataset_base_name}-{train_classes['name']}"

        model_onnx = f"{models_dir}/{train_name}.onnx"
        model_har = f"{models_dir}/{train_name}.har"
        model_hef = f"{models_dir}/{train_name}.hef"

        if os.getenv('TRAIN_STEP_PARSE'):
            parse(model_onnx, model_har, train_classes['class_count'])

        if os.getenv('TRAIN_STEP_COMPILE'):
            if not os.path.exists(model_hef):
                build(model_har, model_hef, train_classes['class_count'])

    for train_classes in trains_classes:
        train_name = f"{train_dataset_base_name}-{train_classes['name']}"
        model_pt = f"{models_dir}/{train_name}.pt"
        model_hef = f"{models_dir}/{train_name}.hef"

        if os.getenv('TRAIN_STEP_GIT'):
            if os.path.exists(model_pt) and os.path.exists(model_hef):
                commit(train_name, model_pt, model_hef)

    purge(f'{train_dataset_path}/train', '*.npy')
    purge(f'{train_dataset_path}/val', '*.npy')
    purge(f'{train_dataset_path}/test', '*.npy')

    print(f"End at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
