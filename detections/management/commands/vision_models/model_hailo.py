import os

import cv2
import numpy as np
from hailo_platform import (
    HEF,
    ConfigureParams,
    FormatType,
    HailoStreamInterface,
    InferVStreams,
    InputVStreamParams,
    OutputVStreamParams,
    VDevice,
)

from detections.management.commands.vision_models.model import Model


class Model_Hailo(Model):

    def __init__(self):
        super().__init__()

        self.target = VDevice()

        # self.model = self.target.create_infer_model(hef_path)
        # self.model = self.model.configure()
        # print(self.model)

        # self.hef_path = hef_path
        # self.hef = HEF(hef_path)
        # #self.model = InferModel(InferModel, hef_path)
        # self.target.configure(self.hef)
        # print(self.target.get_physical_devices())
        # print(self.target.loaded_network_groups)
        # #self.control = Control(self.target.get_physical_devices()[0])
        # self.model = model.configure()
        # self.target.release()

        # The target can be used as a context manager ("with" statement) to ensure it's released on time.
        # Here it's avoided for the sake of simplicity
        # target = VDevice()

        # Loading compiled HEFs to device:
        hef = HEF(os.getenv('MODEL_PATH'))

        # Configure network groups
        configure_params = ConfigureParams.create_from_hef(hef=hef, interface=HailoStreamInterface.PCIe)
        network_groups = self.target.configure(hef, configure_params)
        network_group = network_groups[0]
        network_group_params = network_group.create_params()

        # Create input and output virtual streams params
        input_vstreams_params = InputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
        # output_vstreams_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
        output_vstreams_params = OutputVStreamParams.make(network_group, format_type=FormatType.UINT8)

        # Define dataset params
        input_vstream_info = hef.get_input_vstream_infos()[0]
        output_vstream_info = hef.get_output_vstream_infos()[0]

        self.network_group = network_group
        self.network_group_params = network_group_params
        self.input_vstream_info = input_vstream_info
        self.input_vstreams_params = input_vstreams_params
        self.output_vstream_info = output_vstream_info
        self.output_vstreams_params = output_vstreams_params

        print(network_group)
        print(network_group_params)
        print(input_vstream_info)
        print(input_vstreams_params)
        print(output_vstream_info)
        print(output_vstreams_params)

    def infer(self, frame: cv2.typing.MatLike):
        image_height, image_width, channels = self.input_vstream_info.shape  # 1024, 1024, 3

        frame_copy = frame.copy()
        frame_copy.resize((image_width, image_height))

        # nparr = np.frombuffer(base64.b64decode(frame), np.uint8)
        # img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Decode as color image
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        gray_img_3channel = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)  # Convert back to 3 channels

        image_array = np.array(gray_img_3channel).astype(np.float32) / 255.0  # Normalisation
        image_array = np.transpose(image_array, (2, 0, 1))  # Convertir en format CHW
        # image_array = np.transpose(image_array, [0, 3, 1, 2])  # Convertir en format CHW
        image_array = np.expand_dims(image_array, axis=0)

        # Infer
        with InferVStreams(self.network_group, self.input_vstreams_params,
                           self.output_vstreams_params) as infer_pipeline:
            input_data = {self.input_vstream_info.name: image_array}
            with self.network_group.activate(self.network_group_params):
                infer_results = infer_pipeline.infer(input_data)
                # The result output tensor is infer_results[output_vstream_info.name]
                print(f"Stream output shape is {infer_results[self.output_vstream_info.name].shape}")

        self.target.release()
        return list()
