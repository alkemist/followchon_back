import math

import cv2
from loguru import logger
from ultralytics import YOLO

from detections.management.commands.vision_models.model import Model
from detections.management.commands.vision_models.result_yolo import Result_yolo
from detections.management.commands.vision_models.source import Source
from detections.management.commands.vision_models.supervisor import Supervisor


class Model_YOLO(Model):

    def __init__(self, supervisor: Supervisor, source: Source = Source.VISION):
        super().__init__(supervisor, 'pt', source)

    def check_model(self, origin: str):
        self.supervisor.fill_params()

        if self.model is None or not self.supervisor.current_model_version != self.supervisor.model_version:
            if self.supervisor.current_model_version != self.supervisor.model_version:
                logger.info(f'Load model version "{self.supervisor.model_version}" : {origin}')

            self.supervisor.current_model_version = self.supervisor.model_version

            self.model = YOLO(self.get_model_path(), task='detect')

    def infer(self, frame: cv2.typing.MatLike):
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
