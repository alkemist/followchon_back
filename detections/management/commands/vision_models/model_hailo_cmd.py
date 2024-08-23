import os
import time

import cv2
import numpy as np
from PIL import Image

from detections.management.commands.vision_models.capture_analyse import Capture_analyse
from detections.management.commands.vision_models.hailo_inference import HailoInference
from detections.management.commands.vision_models.model import Model
from detections.management.commands.vision_models.result_yolo import Result_yolo

PADDING_COLOR = (114, 114, 114)


def preprocess(frame: cv2.typing.MatLike, model_w, model_h):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(img)

    """
    Resize image with unchanged aspect ratio using padding.

    Args:
        image (PIL.Image.Image): Input image.
        model_w (int): Model input width.
        model_h (int): Model input height.

    Returns:
        PIL.Image.Image: Preprocessed and padded image.
    """
    img_h, img_w = image.size
    # Scale image
    scale = min(model_w / img_w, model_h / img_h)
    new_img_w, new_img_h = int(img_w * scale), int(img_h * scale)
    image = image.resize((new_img_w, new_img_h), Image.Resampling.BICUBIC)

    # Create a new padded image
    padded_image = Image.new('RGB', (model_w, model_h), PADDING_COLOR)
    padded_image.paste(image, ((model_w - new_img_w) // 2, (model_h - new_img_h) // 2))
    return padded_image


class Model_Hailo_cmd(Model):

    def __init__(self):
        super().__init__()

        self.hailo_inference = HailoInference(os.getenv('MODEL_PATH'))
        self.height, self.width, _ = self.hailo_inference.get_input_shape()
        self.capture_min_score = float(os.getenv('CAPTURE_MIN_SCORE'))

    def infer(self, frame: cv2.typing.MatLike):
        results = None

        try:
            processed_image = preprocess(frame, self.width, self.height)
            # processed_image = preprocess(frame, width, height)

            # (width, height) = frame.shape[1::-1]
            (height, width) = processed_image.size

            # infer_images = [np.array(processed_image)]
            # raw_detections = self.hailo_inference.run(np.array(infer_images))

            # raw_detections = self.hailo_inference.run(np.array([np.array(processed_image)]))
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
                            score,
                            width,
                            height
                        )

                        yolo_result.import_yolo(
                            bbox[1],
                            bbox[0],
                            bbox[3],
                            bbox[2],
                        )

                        yolo_results.append(yolo_result)

            if len(yolo_results) > 0:
                # processed_image.save(f"static/captures/test/{timezone.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}.jpg")
                # draw = ImageDraw.Draw(processed_image)

                # for result in yolo_results:
                #     draw.rectangle([
                #         (xmin * scale_factor, ymin * scale_factor), (xmax * scale_factor, ymax * scale_factor)], outline=color, width=2)

                print('---------------------------------------------')
                print(raw_detections[0])
                print([' '.join(result.to_array()) for result in yolo_results])

                save_time_elapsed = time.time() - self.save_time

                analyse = Capture_analyse(
                    np.asarray(processed_image),
                    self.last_detections_dict, self.families_dict, self.zones
                )

                image_result = analyse.detect(yolo_results)

                if analyse.is_triggered and save_time_elapsed > 1:
                    analyse.save()
                    self.save_time = time.time()

        except Exception as error:
            self.stop = True
            print("ERROR : ")
            print(error)
            print(results)

        return frame

    def destruct(self):
        self.hailo_inference.release_device()
