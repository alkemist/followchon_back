from datetime import datetime

import PIL.Image
import cv2
import numpy as np
from PIL import Image
from loguru import logger

from detections.management.commands.vision_models.hailo_inference_async import HailoAsyncInference
from detections.management.commands.vision_models.model import Model
from detections.management.commands.vision_models.result_yolo import Result_yolo
from detections.management.commands.vision_models.supervisor import Supervisor
from utils.image import ImageHelper


class Model_Hailo(Model):

    def __init__(self, supervisor: Supervisor):
        super().__init__(supervisor)

        self.height = None
        self.width = None

    def check_model(self, origin: str):
        super().fill_objects()
        self.supervisor.fill_params()

        if self.model is None or self.supervisor.current_model_version != self.supervisor.model_version:
            if self.model is not None:
                self.release()

            # if self.supervisor.current_model_version != self.supervisor.model_version:
            logger.info(f'Load model version "{self.supervisor.model_version}" : {origin}')

            self.supervisor.current_model_version = self.supervisor.model_version

            self.model = HailoAsyncInference(self.supervisor.get_model_path())
            self.height, self.width, _ = self.model.get_input_shape()

    def preprocess(self, image: PIL.Image.Image):
        """
        Resize image with unchanged aspect ratio using padding.

        Args:
            image (PIL.Image.Image): Input image.

        Returns:
            PIL.Image.Image: Preprocessed and padded image.
        """

        width, height = image.size
        padded_image = image.copy()

        background_color = (0, 0, 0)
        padding = (0, 0)

        if width > height:
            padded_image = Image.new(image.mode, (width, width), background_color)
            padding = (0, (width - height) // 2)
        elif height < width:
            padded_image = Image.new(image.mode, (height, height), background_color)
            padding = ((height - width) // 2, 0)

        padded_image.paste(image, padding)

        return (
            padding,
            padded_image.size,
            padded_image.resize((self.width, self.height)),
        )

    def infer(self, frame: cv2.typing.MatLike, frame_count, capture_date: datetime):
        saved = False
        image_pil = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_pil)

        (padding, padded_size, processed_image) = self.preprocess(image_pil.copy())

        (height_resized, width_resized) = processed_image.size

        if self.model is None:
            self.check_model('infer')

        raw_detections_queue = self.model.run(np.array(processed_image))

        # if self.supervisor.log_detections:
        #     logger.info(f"Queue detections : {len(raw_detections_queue)}")

        if len(raw_detections_queue) > 10:
            logger.info(f"Queue size too long : {len(raw_detections_queue)}")

        if len(raw_detections_queue) > 0:
            raw_detections = self.model.remove_last_output_results()

            yolo_results = list()

            if raw_detections is not None and len(raw_detections) > 0:
                # if self.supervisor.log_detections:
                #     logger.info(f"Raw detections: {len(raw_detections)}")

                for i, detection in enumerate(raw_detections):
                    # if self.supervisor.log_detections:
                    #     logger.info(f"Detection: {len(detection)}")

                    for result in detection:
                        bbox, score = result[:4], result[4]

                        if score > 0:
                            if self.supervisor.log_detections:
                                logger.info(
                                    f"Class: {i}, Score: {score}, Result: {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}")

                            if score >= self.supervisor.score_min:
                                yolo_result = Result_yolo(
                                    i,
                                    float(score),
                                    width_resized,
                                    height_resized
                                )

                                yolo_result.import_hailo_without_padding(
                                    padded_size,
                                    padding,
                                    float(bbox[1]),
                                    float(bbox[0]),
                                    float(bbox[3]),
                                    float(bbox[2]),
                                )

                                yolo_results.append(yolo_result)

                (frame, saved) = self.analyze(frame, frame_count, capture_date, yolo_results)
        else:
            # No traitement
            logger.info(f"Queue empty")

        return ImageHelper.resize_with_ratio(frame, self.capture_width, None), saved

    def release(self):
        if self.model is not None:
            self.model.release_device()
            self.model = None

            # logger.info("Model device released")
