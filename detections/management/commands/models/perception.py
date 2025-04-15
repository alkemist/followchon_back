import os
import time
from datetime import datetime

import cv2
from pubsub import pub

from detections.management.commands.models.enums.agent_source import Agent_Source
from detections.management.commands.models.enums.event_source import Event_Source
from detections.management.commands.models.enums.event_type import Event_Type
from detections.management.commands.models.memory import Memory
from detections.management.commands.models.signal import Signal
from detections.models import Capture, Detection


class Perception:
    def __init__(
            self,
            memory: Memory,
            model_version_detect,
            model_version_classify,
            frame: cv2.typing.MatLike,
            vision_date: datetime,
    ):
        self.memory = memory
        self.frame = frame
        self.frame_with_detections = frame.copy()
        self.model_version_detect = model_version_detect
        self.model_version_classify = model_version_classify

        self.is_triggered = False
        self.trigger_time = 0

        self.capture_date = datetime(
            vision_date.year,
            vision_date.month,
            vision_date.day,
            vision_date.hour,
            vision_date.minute,
            vision_date.second,
            vision_date.microsecond + self.memory.frame_saved_count,
        )

    def process(self, signals: list[Signal]):
        # self.send_log('process', f'frame_count : {self.memory.frame_count}')

        self.is_triggered = self.memory.source != Agent_Source.VISION

        for signal in signals:
            if signal.cls in self.memory.families_dict:
                signal.family = self.memory.families_dict[signal.cls]

                if signal.family.is_zoned:
                    for zone in self.memory.zones:
                        if zone.has_point((signal.norm_x_center, signal.norm_y_center)):
                            signal.zone = zone
                            break

                if not self.is_triggered:
                    if signal.family.is_tracked:
                        first_detection = signal.family.id not in self.memory.last_detections
                        last_detection = self.memory.last_detections.get(signal.family.id, None)

                        if (
                                first_detection
                                or last_detection is not None and signal.is_move(last_detection,
                                                                                 self.memory.move_tolerance_margin_norm)
                        ):
                            self.memory.last_detections[signal.family.id] = (signal.norm_x_center, signal.norm_y_center)
                            signal.trigger = Detection.Triggers.MOVE

                            self.is_triggered = True

                    if not self.is_triggered:
                        if signal.zone is not None and signal.zone.is_trigger:
                            signal.trigger = Detection.Triggers.ZONE
                        elif signal.family.is_trigger:
                            signal.trigger = Detection.Triggers.FAMILY

                        self.is_triggered = (
                                                    signal.trigger == Detection.Triggers.ZONE or
                                                    signal.trigger == Detection.Triggers.FAMILY
                                            ) and \
                                            (signal.zone is None or not signal.zone.is_ignored)

                self.frame_with_detections = signal.trace(self.frame_with_detections)

            else:
                self.send_log('process', f'unknown family with index "{signal.cls}"')

        if self.is_triggered:
            if os.getenv('ENABLE_SAVE'):
                Capture().write(self.frame, self.capture_date, signals,
                                self.model_version_detect, self.model_version_classify,
                                self.memory.source)

            self.trigger_time = time.time()
            self.memory.frame_saved_count = self.memory.frame_saved_count + 1

    def send_log(self, action: str, infos: str = ''):
        pub.sendMessage(Event_Type.AGENT_LOG, source=Event_Source.PERCEPTION, action=action, infos=infos)
