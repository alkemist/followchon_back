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


class Hailo:

    def __init__(self, hef_path: str):
        # self.capture_width = int(os.getenv('CAPTURE_WIDTH'))

        # self.hef_path = hef_path
        # self.hef = HEF(hef_path)
        # #self.model = InferModel(InferModel, hef_path)
        self.vdevice = VDevice()
        # self.vdevice.configure(self.hef)
        # print(self.vdevice.get_physical_devices())
        # print(self.vdevice.loaded_network_groups)
        # #self.control = Control(self.vdevice.get_physical_devices()[0])
        self.model = self.vdevice.create_infer_model(hef_path)
        # self.model = model.configure()
        # self.vdevice.release()

        print(self.vdevice.loaded_network_groups)

        # The target can be used as a context manager ("with" statement) to ensure it's released on time.
        # Here it's avoided for the sake of simplicity
        target = VDevice()

        # Loading compiled HEFs to device:
        hef = HEF(hef_path)

        # Configure network groups
        configure_params = ConfigureParams.create_from_hef(hef=hef, interface=HailoStreamInterface.PCIe)
        network_groups = target.configure(hef, configure_params)
        network_group = network_groups[0]
        network_group_params = network_group.create_params()

        print(network_group)
        print(network_group_params)
        print(self.model)
        print(self.model.create_params())

        # Create input and output virtual streams params
        # input_vstreams_params = InputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
        input_vstreams_params = InputVStreamParams.make(self.model, format_type=FormatType.FLOAT32)
        # output_vstreams_params = OutputVStreamParams.make(network_group, format_type=FormatType.UINT8)
        output_vstreams_params = OutputVStreamParams.make(self.model, format_type=FormatType.UINT8)

        # Define dataset params
        input_vstream_info = hef.get_input_vstream_infos()[0]
        output_vstream_info = hef.get_output_vstream_infos()[0]

        # Generate random dataset
        # dataset = np.random.randint(low, high, (num_of_images, image_height, image_width, channels)).astype(np.float32)

        # self.network_group = network_group
        # self.network_group_params = network_group_params
        self.input_vstream_info = input_vstream_info
        self.input_vstreams_params = input_vstreams_params
        self.output_vstream_info = output_vstream_info
        self.output_vstreams_params = output_vstreams_params

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
        # with InferVStreams(self.network_group, self.input_vstreams_params,
        with InferVStreams(self.model, self.input_vstreams_params,
                           self.output_vstreams_params) as infer_pipeline:
            input_data = {self.input_vstream_info.name: image_array}
            # with self.network_group.activate(self.network_group_params):
            with self.model.activate(self.model.create_params()):
                infer_results = infer_pipeline.infer(input_data)
                # The result output tensor is infer_results[output_vstream_info.name]
                print(f"Stream output shape is {infer_results[self.output_vstream_info.name].shape}")

        self.vdevice.release()
        return list()
