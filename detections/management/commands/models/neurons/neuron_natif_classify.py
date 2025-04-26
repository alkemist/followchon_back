import os

import cv2
from pubsub import pub
from ultralytics import YOLO

from detections.management.commands.models.enums.event_source import Event_Source
from detections.management.commands.models.enums.event_type import Event_Type
from detections.management.commands.models.enums.log_level import Log_Level
from detections.management.commands.models.neurons.neuron import Neuron
from detections.management.commands.models.tools import get_param


class Neuron_Natif_Classify(Neuron):

    def __init__(self):
        super().__init__('pt')

    def send_log(self, event: str, infos: str = '', level: Log_Level = None):
        pub.sendMessage(Event_Type.AGENT_LOG, source=Event_Source.CLASSIFY, event=event, infos=infos, level=level)

    def check(self, reason: str):
        model_version = int(get_param('vision_model_version_classify'))

        if self.current_model_version is None or self.current_model_version != model_version:
            if self.model is not None:
                self.release()
                
            self.current_model_version = model_version

            self.send_log('load', f"version {self.current_model_version} / {reason}", Log_Level.LOCAL)

            model_path = (f"{os.getenv('MODEL_DIR')}/"
                          f"{os.getenv('MODEL_CLASSIFY_PREFIX')}{self.current_model_version}-chons.{self.model_ext}")

            self.model = YOLO(model_path, task='classify')

    def process(self, frame: cv2.typing.MatLike):
        if frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
            return list()

        if self.model is None:
            self.check('process')

        results = self.model(frame, verbose=False)
        infers = list()
        classes = list()

        for result in results:
            sub_results = result.probs.top5

            for i, index in enumerate(sub_results):
                cls = result.names[index]
                score = result.probs.top5conf.numpy()[i]

                if score >= self.score_min and cls not in classes:
                    classes.append(cls)
                    infers.append([cls, score])

        return infers
