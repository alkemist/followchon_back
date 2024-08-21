# import argparse

import cv2


# import hailo_platform as hailo_platform

# def parse_args():
#     """
#     Initialize argument parser for the script.
#
#     Returns:
#         argparse.Namespace: Parsed arguments.
#     """
#     parser = argparse.ArgumentParser(description="Detection Example")
#     parser.add_argument("-n", "--net", help="Path for the HEF model.", default="yolov7.hef")
#     parser.add_argument("-i", "--input", default="zidane.jpg",
#                         help="Path to the input - either an image or a folder of images.")
#     parser.add_argument("-b", "--batch_size", default=1, type=int, required=False, help="Number of images in one batch")
#     parser.add_argument("-l", "--labels", default="coco.txt",
#                         help="Path to a text file containing labels. If no labels file is provided, coco2017 will be used.")
#
#     return parser.parse_args()


class Hailo:

    def __init__(self, hef_path: str):
        # args = parse_args()
        self.hef_path = hef_path

    def infer(self, frame: cv2.typing.MatLike):
        results = list()
        return results

# import os
#
# import cv2
# import numpy as np
# from hailo_sdk_client import Client
#
#
# class Hailo:
#     def __init__(self, hef_path: str):
#         client = Client()
#         network_group = client.load_hef(hef_path)
#         self.configured_network = client.configure(network_group)
#         self.capture_width = int(os.getenv('CAPTURE_WIDTH'))
#
#     def infer(self, frame: cv2.typing.MatLike):
#         frame = frame.resize((self.capture_width, self.capture_width))
#         image_array = np.array(frame).astype(np.float32) / 255.0  # Normalisation
#         image_array = np.transpose(image_array, (2, 0, 1))  # Convertir en format CHW
#         image_array = np.expand_dims(image_array, axis=0)
#
#         input_tensor = self.configured_network.get_input_tensor()
#         input_tensor.write(image_array)
#
#         output_tensor = self.configured_network.get_output_tensor()
#         output_data = output_tensor.read()
#
#         print(output_data)
#
#         results = list()
#         # {
#         #    boxes: {
#         #       conf: int[]
#         #       cls: int[]
#         #       xyxy: (int, int, int, int)[] # x1, y1, x2, y2
#         #    }[]
#         # }[]
#
#         return results
