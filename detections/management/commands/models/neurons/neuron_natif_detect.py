import math
import os

import cv2
from pubsub import pub
from ultralytics import YOLO

from detections.management.commands.models.enums.event_source import Event_Source
from detections.management.commands.models.enums.event_type import Event_Type
from detections.management.commands.models.neurons.neuron import Neuron
from detections.management.commands.models.signal import Signal
from detections.management.commands.models.tools import get_param


class Neuron_Natif_Detect(Neuron):

    def __init__(self, score_min: float):
        super().__init__('pt', score_min)

    def send_log(self, action: str, infos: str = ''):
        pub.sendMessage(Event_Type.AGENT_LOG, source=Event_Source.DETECT, action=action, infos=infos)

    def check(self, origin: str):
        self.send_log('check', origin)

        model_version = get_param('vision_model_version_detect')

        if self.current_model_version is None or self.current_model_version != model_version:
            self.current_model_version = model_version
            model_path = (f"{os.getenv('MODEL_DIR')}/"
                          f"{os.getenv('MODEL_DETECT_PREFIX')}{self.current_model_version}-all.{self.model_ext}")

            self.model = YOLO(model_path, task='detect')

    def process(self, frame: cv2.typing.MatLike):
        # self.send_log('process')

        if self.model is None:
            self.check('process')

        results = self.model(frame, stream=True, verbose=False)

        (width, height) = frame.shape[1::-1]
        yolo_results = list()

        for result in results:
            for box in result.boxes:
                tl_x, tl_y, br_x, br_y = box.xyxy[0]
                cls = int(box.cls[0])
                score = math.ceil((box.conf[0] * 100)) / 100

                if score >= self.score_min:
                    yolo_result = Signal(
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

        return yolo_results
