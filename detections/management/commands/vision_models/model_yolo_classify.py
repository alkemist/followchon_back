import os

import cv2
import sympy
from loguru import logger
from sympy import mpmath
from ultralytics import YOLO

from detections.management.commands.vision_models.model import Model
from detections.management.commands.vision_models.source import Source
from detections.management.commands.vision_models.supervisor import Supervisor

sympy.__version__
mpmath.__version__


class Model_YOLO_Classify(Model):

    def __init__(self, supervisor: Supervisor):
        super().__init__(supervisor, 'pt', Source.VISION)

    def check_model(self, origin: str):
        self.supervisor.fill_params()

        if self.model is None or not self.current_model_version != self.supervisor.model_version_classify:
            if self.current_model_version != self.supervisor.model_version_classify:
                logger.info(f'Load model version "{self.supervisor.model_version_classify}" : {origin}')

            self.current_model_version = self.supervisor.model_version_classify

            model_path = (f"{os.getenv('MODEL_DIR')}/"
                          f"{os.getenv('MODEL_CLASSIFY_PREFIX')}{self.current_model_version}-chons.{self.model_ext}")

            self.model = YOLO(model_path, task='classify')

    def infer(self, frame: cv2.typing.MatLike):
        results = self.model(frame, stream=True, verbose=False)

        for result in results:
            cls = result.names[result.probs.top1]
            score = result.probs.data.numpy()[0]

            if score >= self.supervisor.score_min:
                return (cls, score)

        return None
