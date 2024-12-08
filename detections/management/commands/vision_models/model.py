import os

import cv2

from detections.management.commands.vision_models.source import Source
from detections.management.commands.vision_models.supervisor import Supervisor
from detections.management.commands.vision_models.type import Type


class Model:

    def __init__(self, supervisor: Supervisor, model_ext: str = 'pt',
                 source=Source.VISION):
        self.supervisor = supervisor
        self.model = None
        self.model_ext = model_ext
        self.source = source
        self.current_model_version = None

    def get_model_path(self):
        return (f"{os.getenv('MODEL_DIR')}/"
                f"{os.getenv('MODEL_PREFIX')}{self.current_model_version}.{self.model_ext}")

    def get_model_path_double(self, model_type: Type = Type.ALL):
        return (f"{os.getenv('MODEL_DIR')}/"
                f"{os.getenv('MODEL_PREFIX')}{self.current_model_version}-{model_type}.{self.model_ext}")

    def infer(self, frame: cv2.typing.MatLike):
        raise Exception('Infer not implemented')

    def check_model(self, origin: str):
        raise Exception('Check model not implemented')

    def release(self):
        return False
