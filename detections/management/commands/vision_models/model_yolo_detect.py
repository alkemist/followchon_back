import math
import os

import cv2
from loguru import logger
from ultralytics import YOLO

from detections.management.commands.vision_models.model import Model
from detections.management.commands.vision_models.result_yolo import Result_yolo
from detections.management.commands.vision_models.source import Source
from detections.management.commands.vision_models.supervisor import Supervisor


class Model_YOLO_Detect(Model):

    def __init__(self, supervisor: Supervisor, source: Source = Source.VISION):
        super().__init__(supervisor, 'pt', source)

    def check_model(self, origin: str):
        self.supervisor.fill_params()

        if self.model is None or not self.current_model_version != self.supervisor.model_version_detect:
            if self.current_model_version != self.supervisor.model_version_detect:
                logger.info(f'Load model version "{self.supervisor.model_version_detect}" : {origin}')

            self.current_model_version = self.supervisor.model_version_detect

            model_path = (f"{os.getenv('MODEL_DIR')}/"
                          f"{os.getenv('MODEL_DETECT_PREFIX')}{self.current_model_version}-all.{self.model_ext}")

            self.model = YOLO(model_path, task='detect')

    def infer(self, frame: cv2.typing.MatLike):
        if self.model is None:
            self.check_model('infer')

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
                        math.floor(tl_x),
                        math.floor(tl_y),
                        math.ceil(br_x),
                        math.ceil(br_y),
                    )

                    yolo_results.append(yolo_result)

                    # image_pil = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # image_pil = Image.fromarray(image_pil)
                    # draw = ImageDraw.Draw(image_pil)
                    #
                    #
                    # print(float(tl_x), float(tl_y), float(br_x), float(br_y))
                    # print([(yolo_result.norm_x_center, yolo_result.norm_y_center),
                    #        (yolo_result.norm_width, yolo_result.norm_height)])
                    #
                    # draw.rectangle([(yolo_result.ortho_tl_x, yolo_result.ortho_tl_y),
                    #                 (yolo_result.ortho_br_x, yolo_result.ortho_br_y)], width=2)
                    # image_pil.save(f"{capture_date.strftime('%Y-%m-%d_%H-%M-%S-%f')}.jpg", 'JPEG')
                    #
                    # exit()

        return yolo_results
