import numpy as np
import cv2

from hailo_platform.pyhailort.pyhailort import HEF, InferModel, Control, Device, VDevice

class Hailo:

    def __init__(self, hef_path: str):
        self.hef_path = hef_path
        self.hef = HEF(hef_path)
        #self.model = InferModel(InferModel, hef_path)
        self.vdevice = VDevice()
        self.vdevice.configure(self.hef)
        print(self.vdevice.get_physical_devices())
        print(self.vdevice.loaded_network_groups)
        #self.control = Control(self.vdevice.get_physical_devices()[0])
        model = self.vdevice.create_infer_model(self.hef_path)
        self.model = model.configure()
        self.vdevice.release()

    def infer(self, frame: cv2.typing.MatLike):
        frame = frame.resize((self.capture_width, self.capture_width))
        image_array = np.array(frame).astype(np.float32) / 255.0  # Normalisation
        image_array = np.transpose(image_array, (2, 0, 1))  # Convertir en format CHW
        image_array = np.expand_dims(image_array, axis=0)

        results = list()
        return results

