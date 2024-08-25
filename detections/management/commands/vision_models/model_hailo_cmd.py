import os
import time

import PIL.Image
import cv2
import numpy as np
from PIL import Image, ImageDraw
from django.utils import timezone

from detections.management.commands.vision_models.capture_analyse import Capture_analyse
from detections.management.commands.vision_models.hailo_inference import HailoInference
from detections.management.commands.vision_models.model import Model
from detections.management.commands.vision_models.result_yolo import Result_yolo

PADDING_COLOR = (114, 114, 114)


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


def preprocess(image: PIL.Image.Image, model_w, model_h):
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

    if width > height:
        padded_image = Image.new(image.mode, (width, width), background_color)
        padded_image.paste(image, (0, (width - height) // 2))
    elif height < width:
        padded_image = Image.new(image.mode, (height, height), background_color)
        padded_image.paste(image, ((height - width) // 2, 0))

    return padded_image.resize((model_w, model_h))


class Model_Hailo_cmd(Model):

    def __init__(self):
        super().__init__()

        self.hailo_inference = HailoInference(os.getenv('MODEL_PATH'))
        self.height, self.width, _ = self.hailo_inference.get_input_shape()
        self.capture_min_score = float(os.getenv('CAPTURE_MIN_SCORE'))

    def infer(self, frame: cv2.typing.MatLike):
        results = None

        try:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

            processed_image = preprocess(image, self.width, self.height)

            (width_original, height_original) = frame.shape[1::-1]
            (height_resized, width_resized) = processed_image.size

            raw_detections = self.hailo_inference.run(np.array(processed_image))

            yolo_results = list()

            for i, detection in enumerate(raw_detections[0]):
                if len(detection) == 0:
                    continue

                for det in detection:
                    bbox, score = det[:4], det[4]

                    if score >= self.capture_min_score:
                        yolo_result = Result_yolo(
                            i,
                            float(score),
                            width_resized,
                            height_resized
                        )

                        yolo_result.import_hailo(
                            float(bbox[1]),
                            float(bbox[0]),
                            float(bbox[3]),
                            float(bbox[2]),
                        )

                        yolo_results.append(yolo_result)

            if len(yolo_results) > 0:
                save_time_elapsed = time.time() - self.save_time

                analyse = Capture_analyse(
                    np.asarray(processed_image),
                    self.last_detections_dict, self.families_dict, self.zones
                )

                image_result = analyse.detect(yolo_results)

                if analyse.is_triggered and save_time_elapsed > 1:
                    f_name = timezone.now().strftime('%Y-%m-%d_%H-%M-%S-%f')

                    print('---------------------------------------------')
                    print(raw_detections[0])
                    print([' '.join(result.to_array()) for result in yolo_results])

                    draw = ImageDraw.Draw(processed_image)

                    for result in yolo_results:
                        draw.rectangle(
                            [
                                # (result.bbox[0] * result.ref_width, result.bbox[1] * result.ref_height),
                                # (result.bbox[2] * result.ref_width, result.bbox[3] * result.ref_height),
                                (result.ortho_tl_x, result.ortho_tl_y),
                                (result.ortho_br_x, result.ortho_br_y),
                            ],
                            outline=255,
                            width=2
                        )

                    processed_image.save(f"static/captures/test/{f_name}.jpg")

                    # analyse.save()
                    self.save_time = time.time()

        except Exception as error:
            self.stop = True
            print("ERROR : ")
            print(error)
            print(results)

        return frame

    def destruct(self):
        self.hailo_inference.release_device()
