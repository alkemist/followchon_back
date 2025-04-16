import os

import PIL.Image
import cv2
import numpy as np
from PIL import Image
from hailo_platform import (VDevice, HailoSchedulingAlgorithm)
from pubsub import pub

from detections.management.commands.models.enums.event_source import Event_Source
from detections.management.commands.models.enums.event_type import Event_Type
from detections.management.commands.models.enums.log_level import Log_Level
from detections.management.commands.models.neurons.hailo_inference_async import HailoAsyncInference
from detections.management.commands.models.neurons.neuron import Neuron
from detections.management.commands.models.signal import Signal
from detections.management.commands.models.tools import get_param


class Neuron_Hailo_Detect(Neuron):

    def __init__(self, score_min: float):
        super().__init__('hef', score_min)

        self.height = None
        self.width = None

    def send_log(self, event: str, infos: str = '', level: Log_Level = None):
        pub.sendMessage(Event_Type.AGENT_LOG, source=Event_Source.DETECT, event=event, infos=infos, level=level)

    def check(self, reason: str):

        model_version = get_param('vision_model_version_detect')

        if self.model is None or self.current_model_version is None or self.current_model_version != model_version:
            self.current_model_version = model_version

            self.send_log('load', f"version {self.current_model_version} / {reason}")

            model_path = (f"{os.getenv('MODEL_DIR')}/"
                          f"{os.getenv('MODEL_DETECT_PREFIX')}{self.current_model_version}-all.{self.model_ext}")

            params = VDevice.create_params()
            params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
            vdevice = VDevice(params)

            self.model = HailoAsyncInference(vdevice, model_path)
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

    def process(self, frame: cv2.typing.MatLike):
        # self.send_log('process')

        image_pil = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_pil)

        (padding, padded_size, processed_image) = self.preprocess(image_pil.copy())

        (height_resized, width_resized) = processed_image.size

        yolo_results = list()

        if self.model is None:
            self.check('process')

        raw_detections_queue = self.model.run(np.array(processed_image))

        if len(raw_detections_queue) > 10:
            self.send_log("process", f"Queue size too long : {len(raw_detections_queue)}")

        if self.model and len(raw_detections_queue) > 0:
            raw_detections = self.model.remove_last_output_results()

            if raw_detections is not None and len(raw_detections) > 0:

                for i, detection in enumerate(raw_detections):

                    for result in detection:
                        bbox, score = result[:4], result[4]

                        if score > 0:
                            if score >= self.score_min:
                                yolo_result = Signal(
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
                                    frame.shape[1],
                                    frame.shape[0],
                                )

                                yolo_results.append(yolo_result)
        else:
            # No traitement
            self.send_log("process", "Queue empty")

        return yolo_results

    def release(self):
        if self.model is not None:
            self.model.release_device()
            self.model = None
