import os
import time

import cv2
from ultralytics import YOLO

from detections.management.commands.vision_models.capture_analyse import Capture_analyse
from detections.management.commands.vision_models.model import Model
from helpers.image import ImageHelper


class Model_YOLO(Model):

    def __init__(self):
        super().__init__()
        self.model = YOLO(os.getenv('MODEL_PATH'))

    def infer(self, frame: cv2.typing.MatLike):
        # if self.hailo_enabled:
        #     results = cast(self.model, Hailo).infer(self.model, frame)
        # else:
        results = self.model(frame, stream=True, verbose=False)

        save_time_elapsed = time.time() - self.save_time

        analyse = Capture_analyse(frame, self.last_detections_dict, self.families_dict, self.zones)
        frame = analyse.detect(results)

        if analyse.is_triggered and save_time_elapsed > 1:
            analyse.save()
            self.save_time = time.time()

        return ImageHelper.resize_with_ratio(frame, self.capture_width, None)
