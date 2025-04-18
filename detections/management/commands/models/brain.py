import time
from datetime import datetime

import cv2
from pubsub import pub

from detections.management.commands.models.enums.agent_source import Agent_Source
from detections.management.commands.models.enums.architecture import Architecture
from detections.management.commands.models.enums.event_source import Event_Source
from detections.management.commands.models.enums.event_type import Event_Type
from detections.management.commands.models.enums.log_level import Log_Level
from detections.management.commands.models.memory import Memory
from detections.management.commands.models.neurons.neuron_natif_classify import Neuron_Natif_Classify
from detections.management.commands.models.neurons.neuron_natif_detect import Neuron_Natif_Detect
from detections.management.commands.models.perception import Perception


# Il s'occupe de traiter les images
class Brain:
    def __init__(
            self,
            archi: Architecture,
            memory: Memory
    ):
        self.send_log('init', '', Log_Level.LOCAL)

        pub.subscribe(self.process_neurons, Event_Type.BRAIN_PROCESS)

        self.archi = archi
        self.memory = memory

        self.enabled = False
        self.records_count = 0
        self.last_capture_seconds = 0

        self.neuron_classify = None
        self.neuron_detect = None

        self.perception = None

    def send_log(self, event: str, infos: str = '', level: Log_Level = None):
        pub.sendMessage(Event_Type.AGENT_LOG, source=Event_Source.BRAIN, event=event, infos=infos, level=level)

    def check(self, reason: str):
        self.memory.check(reason)
        self.neuron_classify.check(reason)
        self.neuron_detect.check(reason)

    def process_neurons(self, frame: cv2.typing.MatLike, vision_date: datetime):
        self.memory.perception = Perception(
            self.memory,
            self.neuron_detect.current_model_version,
            self.neuron_classify.current_model_version,
            frame,
            vision_date,
        )

        detect_signals = self.neuron_detect.process(frame)
        signals = detect_signals.copy()

        if self.memory.source == Agent_Source.VISION:
            detect_safes_cls = list()
            detect_unsafes_by_family = {}

            for detect_result in detect_signals:

                image_result = frame[
                               detect_result.ortho_tl_y:detect_result.ortho_br_y,
                               detect_result.ortho_tl_x:detect_result.ortho_br_x
                               ]
                classify_signals = self.neuron_classify.process(image_result)

                for [slug, score] in classify_signals:
                    if slug in self.memory.classify_families_dict:
                        family = self.memory.classify_families_dict[slug]
                        classify_yolo_result = detect_result.clone(family.index, score)

                        if not family.is_unique:
                            detect_safes_cls.append(family.index)
                            signals.append(classify_yolo_result)
                        else:
                            if family.index not in detect_unsafes_by_family:
                                detect_unsafes_by_family[family.index] = []

                            detect_unsafes_by_family[family.index].append(classify_yolo_result)
                    else:
                        self.send_log('process_neurons', f'unknown family with slug "{slug}"')

            for cls, detect_unsafes in detect_unsafes_by_family.items():
                detect_unsafes = sorted(detect_unsafes, key=lambda result: result.score, reverse=True)

                if cls not in detect_safes_cls:
                    detect_safes_cls.append(cls)
                    signals.append(detect_unsafes[0])

        signals = sorted(signals, key=lambda signal: signal.cls)

        self.memory.perception.process(signals)

    def start(self):
        self.send_log('start', '', Log_Level.LOCAL)

        self.neuron_classify = Neuron_Natif_Classify(self.memory.score_min)

        if self.archi == Architecture.HAILO:
            from detections.management.commands.models.neurons.neuron_hailo_detect import Neuron_Hailo_Detect
            self.neuron_detect = Neuron_Hailo_Detect(self.memory.score_min)
        else:
            self.neuron_detect = Neuron_Natif_Detect(self.memory.score_min)

    def sleep(self, time_minutes):
        self.send_log('sleep', Log_Level.LOCAL)

        self.neuron_detect.release()
        time.sleep(time_minutes * 60)
        self.memory.last_record_seconds = time.time()
        self.send_log('end sleep', Log_Level.LOCAL)
        self.check('sleep')

    def stop(self):
        self.send_log('stop', '', Log_Level.LOCAL)

        self.neuron_classify.release()
        self.neuron_detect.release()
