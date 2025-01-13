import os

import cv2
from loguru import logger
from ultralytics import YOLO

from detections.management.commands.vision_models.model import Model
from detections.management.commands.vision_models.source import Source
from detections.management.commands.vision_models.supervisor import Supervisor


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
        if frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
            return list()

        if self.model is None:
            self.check_model('infer')

        results = self.model(frame, verbose=False)
        infers = list()
        classes = list('guinea-pig')

        for result in results:
            sub_results = result.probs.top5

            for i, index in enumerate(sub_results):
                cls = result.names[index]
                score = result.probs.top5conf.numpy()[i]

                if score >= self.supervisor.score_min and cls not in classes:
                    classes.append(cls)
                    infers.append([cls, score])

        return infers
