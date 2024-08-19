# script_quantize.py
import sys

from onnxruntime.quantization import quantize_dynamic, QuantType, quant_pre_process
from ultralytics import YOLO


# from torch.ao.quantization import quantize_dynamic


def quantize_model(model_path):
    yolo_model_path = f"{model_path}.pt"
    har_model_path = f"{model_path}.har"
    onnx_model_path = f"{model_path}.onnx"
    onnx_preprocessed_model_path = f"{model_path}_preprocessed.onnx"
    har_preprocessed_model_path = f"{model_path}_preprocessed.har"
    onnx_quantized_model_path = f"{model_path}_quantized.onnx"
    har_quantized_model_path = f"{model_path}_quantized.har"

    # Load the YOLOv8 model
    ul_model = YOLO(yolo_model_path)
    print(f'Model yolo loaded {yolo_model_path}')

    ul_model.export(format="onnx")
    print(f'Model onnx quantized saved in {onnx_model_path}')

    # Pre-processing prepares a float32 model for quantization.
    quant_pre_process(input_model_path=onnx_model_path, output_model_path=onnx_preprocessed_model_path)
    print(f'Model onnx pre-processed in {har_preprocessed_model_path}')

    # Quantifier dynamiquement le modèle (c'est souvent la méthode la plus simple)
    quantize_dynamic(model_input=onnx_preprocessed_model_path,
                     model_output=onnx_quantized_model_path,
                     per_channel=False,  # Adjust as needed
                     weight_type=QuantType.QUInt8)
    print(f'Model onnx quantized in {onnx_quantized_model_path}')


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python quantize.py <input_model_path_without_extension>")
        sys.exit(1)

    input_model_path = sys.argv[1]

    quantize_model(input_model_path)
