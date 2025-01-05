import cv2

from detections.management.commands.vision_models.source import Source
from detections.management.commands.vision_models.supervisor import Supervisor


class Model:

    def __init__(self, supervisor: Supervisor, model_ext: str = 'pt',
                 source=Source.VISION):
        self.supervisor = supervisor
        self.model = None
        self.model_ext = model_ext
        self.source = source
        self.current_model_version = None
        self.current_model_version = None

    def infer(self, frame: cv2.typing.MatLike):
        raise Exception('Infer not implemented')

    def check_model(self, origin: str):
        raise Exception('Check model not implemented')

    def release(self):
        return False
