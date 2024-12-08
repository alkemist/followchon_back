import torch
from onnx import __version__, IR_VERSION
from onnx.defs import onnx_opset_version
from ultralytics import YOLO

from scripts.combine import CombinedModel

print("PyTorch version :", torch.__version__)
print("CUDA Devices :", torch.cuda.device_count())
print(f"ONNX version={__version__!r}, opset={onnx_opset_version()}, ir_version={IR_VERSION}")

train_types = ['all', 'chons']
train_classes = [[0], [1, 2]]

models = [
    f"../models/guinea-pig-8n-v40-{train_type}.pt"
    for train_type in train_types
]

combined_model = CombinedModel(
    [YOLO(model) for model in models]
)

combined_model.export('../models/guinea-pig-8n-v40.onnx')
