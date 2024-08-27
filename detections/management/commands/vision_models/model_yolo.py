import math
import os
import time

import cv2
from ultralytics import YOLO

from detections.management.commands.vision_models.capture_analyse import Capture_analyse
from detections.management.commands.vision_models.model import Model
from detections.management.commands.vision_models.result_yolo import Result_yolo
from helpers.image import ImageHelper


class Model_YOLO(Model):

    def __init__(self):
        super().__init__()

        self.model = YOLO(os.getenv('MODEL_PATH'))
        self.capture_min_score = float(os.getenv('CAPTURE_MIN_SCORE'))  # 0.03 < > 0.02

    def infer(self, frame: cv2.typing.MatLike):
        results = self.model(frame, stream=True, verbose=False)

        (width, height) = frame.shape[1::-1]
        yolo_results = list()

        for result in results:
            for box in result.boxes:
                tl_x, tl_y, br_x, br_y = box.xyxy[0]
                cls = int(box.cls[0])
                score = math.ceil((box.conf[0] * 100)) / 100

                if score >= self.capture_min_score:
                    yolo_result = Result_yolo(
                        cls,
                        score,
                        width,
                        height
                    )

                    yolo_result.import_ortho(
                        float(tl_x),
                        float(tl_y),
                        float(br_x),
                        float(br_y),
                    )

                    yolo_results.append(yolo_result)

        if len(yolo_results) > 0:
            save_time_elapsed = time.time() - self.save_time

            analyse = Capture_analyse(frame, self.last_detections_dict, self.families_dict, self.zones)
            frame = analyse.detect(yolo_results)

            if analyse.is_triggered and save_time_elapsed > 1:
                analyse.save()
                self.save_time = time.time()

        return ImageHelper.resize_with_ratio(frame, self.capture_width, None)
