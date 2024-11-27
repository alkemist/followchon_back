import os
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

train_previous_path = os.getenv('TRAIN_MODEL_PATH')
train_dataset_path = os.getenv('TRAIN_DATASET_PATH')
train_dataset_yaml_path = f"{train_dataset_path}/data.yaml"
train_dataset_name = os.getenv('TRAIN_DATASET_NAME')
train_device = os.getenv('TRAIN_DEVICE')
train_resume = os.getenv('TRAIN_RESUME')
hailo_sdk_version = os.getenv('TRAIN_SDK_VERSION')
model_base_version = os.getenv('TRAIN_MODEL_BASE_VERSION')
model_nms_version = os.getenv('TRAIN_MODEL_NMS_VERSION')
train_dataset_classes = os.getenv('TRAIN_CLASSES')
train_calib_dir = os.getenv('TRAIN_CALIB_DIR')

model_train_last = f"{runs_dir}/{train_dataset_name}/weights/best.pt"
model_pt = f"{models_dir}/{train_dataset_name}.pt"
model_onnx = f"{models_dir}/{train_dataset_name}.onnx"
model_har = f"{models_dir}/{train_dataset_name}.har"
model_hef = f"{models_dir}/{train_dataset_name}.hef"

end_node_names = (
    '/model.22/cv2.0/cv2.0.2/Conv',
    '/model.22/cv3.0/cv3.0.2/Conv',
    '/model.22/cv2.1/cv2.1.2/Conv',
    '/model.22/cv3.1/cv3.1.2/Conv',
    '/model.22/cv2.2/cv2.2.2/Conv',
    '/model.22/cv3.2/cv3.2.2/Conv'
)


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


def parse():
    command_parse = (
        "docker", "exec", "-i", f"hailo_ai_sw_suite_{hailo_sdk_version}_container",
        "hailomz", "parse",
        "--hw-arch", "hailo8l",
        "--ckpt", f"/local/shared_with_docker/followchon_back/{model_onnx}",
        # "--start-node-names", "images",
        # "--end-node-names") + end_node_names + (
        "--yaml",
        f"/local/shared_with_docker/followchon_back/models/config/{model_base_version}/hef_config_n_{model_nms_version}.yaml",
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


def build():
    command_compile = (
        "docker", "exec", "-i", f"hailo_ai_sw_suite_{hailo_sdk_version}_container",
        "hailomz", "compile",
        "--hw-arch", "hailo8l",
        "--har", f"/local/shared_with_docker/followchon_back/{model_har}",
        # "--ckpt", f"/local/shared_with_docker/followchon_back/{model_onnx}",
        "--classes", train_dataset_classes,
        # "--start-node-names", "images",
        # "--end-node-names") + end_node_names + (
        "--calib-path", f"/local/shared_with_docker/followchon_back/{train_dataset_path}/{train_calib_dir}",
        # "--model-script",
        # f"/local/shared_with_docker/followchon_back/models/config/{model_base_version}/yolo.alls",
        "--yaml",
        f"/local/shared_with_docker/followchon_back/models/config/{model_base_version}/hef_config_n_{model_nms_version}.yaml",
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


def commit():
    for path in execute(('git', 'pull')):
        print(path, end="")

    for path in execute(('git', 'add', model_pt, model_hef)):
        print(path, end="")

    for path in execute(('git', 'commit', '-m', train_dataset_name)):
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

    if os.getenv('TRAIN_STEP_TRAIN'):
        train()

    if os.getenv('TRAIN_STEP_EXPORT'):
        export()

    if os.getenv('TRAIN_STEP_PARSE'):
        parse()

    if os.getenv('TRAIN_STEP_COMPILE'):
        build()

    if os.getenv('TRAIN_STEP_GIT'):
        commit()

    print(f"End at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
