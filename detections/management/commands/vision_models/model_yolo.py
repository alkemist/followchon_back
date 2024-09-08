import math

import cv2
from loguru import logger
from ultralytics import YOLO

from detections.management.commands.vision_models.model import Model
from detections.management.commands.vision_models.result_yolo import Result_yolo
from detections.management.commands.vision_models.supervisor import Supervisor
from helpers.image import ImageHelper


class Model_YOLO(Model):

    def __init__(self, supervisor: Supervisor):
        super().__init__(supervisor)

    def check_model(self):
        super().fill_objects()
        self.supervisor.fill_params()

        if self.model is None or not self.supervisor.current_model_version != self.supervisor.model_version:
            if self.supervisor.current_model_version != self.supervisor.model_version:
                logger.info(f'Load model version "{self.supervisor.current_model_version}"')

            self.supervisor.current_model_version = self.supervisor.model_version

            self.model = YOLO(self.supervisor.get_model_path(), task='detect')

    def infer(self, frame: cv2.typing.MatLike, frame_count, datestr):
        results = self.model(frame, stream=True, verbose=False)

        (width, height) = frame.shape[1::-1]
        yolo_results = list()

        for result in results:
            for box in result.boxes:
                tl_x, tl_y, br_x, br_y = box.xyxy[0]
                cls = int(box.cls[0])
                score = math.ceil((box.conf[0] * 100)) / 100

                if score >= self.supervisor.score_min:
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

        (frame, saved) = self.analyze(frame, frame_count, datestr, yolo_results)

        return ImageHelper.resize_with_ratio(frame, self.capture_width, None), saved
