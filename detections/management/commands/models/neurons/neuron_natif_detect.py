import math
import os

import cv2
from pubsub import pub
from ultralytics import YOLO

from detections.management.commands.models.enums.event_source import Event_Source
from detections.management.commands.models.enums.event_type import Event_Type
from detections.management.commands.models.enums.log_level import Log_Level
from detections.management.commands.models.neurons.neuron import Neuron
from detections.management.commands.models.signal import Signal
from detections.management.commands.models.tools import get_param


class Neuron_Natif_Detect(Neuron):

    def __init__(self):
        super().__init__('pt')

    def send_log(self, event: str, infos: str = '', level: Log_Level = None):
        pub.sendMessage(Event_Type.AGENT_LOG, source=Event_Source.DETECT, event=event, infos=infos, level=level)

    def check(self, reason: str):
        model_version = int(get_param('vision_model_version_detect'))

        if self.current_model_version is None or self.current_model_version != model_version:
            if self.model is not None:
                self.release()
                
            self.current_model_version = model_version

            self.send_log('load', f"version {self.current_model_version} / {reason}", Log_Level.LOCAL)

            model_path = (f"{os.getenv('MODEL_DIR')}/"
                          f"{os.getenv('MODEL_DETECT_PREFIX')}{self.current_model_version}-all.{self.model_ext}")

            self.model = YOLO(model_path, task='detect')

    def process(self, frame: cv2.typing.MatLike):
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
