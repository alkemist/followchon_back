import os

import cv2
import numpy as np
from hailo_sdk_client import Client


class Hailo:
    def __init__(self, hef_path: str):
        client = Client()
        network_group = client.load_hef(hef_path)
        self.configured_network = client.configure(network_group)
        self.capture_width = int(os.getenv('CAPTURE_WIDTH'))

    def infer(self, frame: cv2.typing.MatLike):
        frame = frame.resize((self.capture_width, self.capture_width))
        image_array = np.array(frame).astype(np.float32) / 255.0  # Normalisation
        image_array = np.transpose(image_array, (2, 0, 1))  # Convertir en format CHW
        image_array = np.expand_dims(image_array, axis=0)

        input_tensor = self.configured_network.get_input_tensor()
        input_tensor.write(image_array)

        output_tensor = self.configured_network.get_output_tensor()
        output_data = output_tensor.read()

        print(output_data)

        results = list()
        # {
        #    boxes: {
        #       conf: int[]
        #       cls: int[]
        #       xyxy: (int, int, int, int)[] # x1, y1, x2, y2
        #    }[]
        # }[]

        return results
