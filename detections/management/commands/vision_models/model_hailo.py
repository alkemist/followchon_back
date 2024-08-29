import time

import PIL.Image
import cv2
import numpy as np
from PIL import Image
from loguru import logger

from detections.management.commands.vision_models.capture_analyse import Capture_analyse
from detections.management.commands.vision_models.hailo_inference_async import HailoAsyncInference
from detections.management.commands.vision_models.model import Model
from detections.management.commands.vision_models.result_yolo import Result_yolo
from helpers.image import ImageHelper


class Model_Hailo(Model):

    def __init__(self):
        super().__init__()

        self.height = None
        self.width = None

        self.check_model()

    def check_model(self):
        super().fill_objects()
        super().fill_params()

        if self.model is None or self.current_model_version != self.model_version:
            if self.model is not None:
                self.release()

            self.current_model_version = self.model_version

            logger.info(f'Load model version "{self.current_model_version}"')

            self.model = HailoAsyncInference(super().get_model_path())
            self.height, self.width, _ = self.model.get_input_shape()

    def preprocess(self, image: PIL.Image.Image):
        """
        Resize image with unchanged aspect ratio using padding.

        Args:
            image (PIL.Image.Image): Input image.
            model_w (int): Model input width.
            model_h (int): Model input height.

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

    def infer(self, frame: cv2.typing.MatLike):
        image_pil = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_pil)

        (padding, padded_size, processed_image) = self.preprocess(image_pil)

        (height_resized, width_resized) = processed_image.size

        raw_detections = self.model.run(np.array(processed_image))

        if len(raw_detections) > 10:
            logger.info(f"Queue size too long : {len(raw_detections)}")

        if len(raw_detections) > 0:
            raw_detection = self.model.remove_last_output_results()

            yolo_results = list()

            if raw_detection is not None and len(raw_detection) > 0:
                for i, detection in enumerate(raw_detection):
                    if len(detection) == 0:
                        continue

                    for det in detection:
                        bbox, score = det[:4], det[4]

                        if score >= self.min_score:
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

                if len(yolo_results) > 0:
                    save_time_elapsed = time.time() - self.save_time

                    analyse = Capture_analyse(
                        frame,
                        self.last_detections_dict, self.families_dict, self.zones
                    )

                    frame = analyse.detect(yolo_results)

                    if analyse.is_triggered and save_time_elapsed > 1:
                        analyse.save()
                        self.save_time = time.time()

        else:
            # No traitement
            logger.info(f"Queue empty")

        return ImageHelper.resize_with_ratio(frame, self.capture_width, None)

    def release(self):
        if self.model is not None:
            self.model.release_device()
            self.model = None

            logger.info("Model device released")
