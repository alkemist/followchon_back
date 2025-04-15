import os
import random
import re
import time
from datetime import datetime
from pathlib import Path
from time import sleep

import cv2
from pubsub import pub

from detections.management.commands.models.enums.agent_source import Agent_Source
from detections.management.commands.models.enums.event_source import Event_Source
from detections.management.commands.models.enums.event_type import Event_Type
from detections.management.commands.models.memory import Memory
from utils.image import ImageHelper


class Eye:
    def __init__(
            self,
            memory: Memory,
    ):
        self.memory = memory

        self.last_frame_seconds = 0

    def send_log(self, action: str, infos: str = ''):
        pub.sendMessage(Event_Type.AGENT_LOG, source=Event_Source.EYE, action=action, infos=infos)

    def watch(self):
        vision_date = datetime.now()
        path = None

        if self.memory.source == Agent_Source.VISION or self.memory.source == Agent_Source.VIDEO:
            path = self.memory.get_last_memory()
            self.memory.check('watch')

            self.send_log('open', path)

            self.memory.frame_saved_count = 0

            if self.memory.source == Agent_Source.VISION:
                self.memory.last_record_seconds = time.time()

                file_date = Path(path).stem
                date_values = re.split('[-_]', file_date)
                vision_date = datetime(
                    int(date_values[0]),
                    int(date_values[1]),
                    int(date_values[2]),
                    int(date_values[3]),
                    int(date_values[4]),
                    int(date_values[5]),
                    random.randint(0, 500)
                )

            cap = cv2.VideoCapture(path)
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            ret = True

            while ret and cap.isOpened() and self.memory.brain_enabled:
                frame_seconds_elapsed = time.time() - self.last_frame_seconds
                ret, frame = cap.read()

                if ret and frame is not None and frame.size > 0:
                    if frame_seconds_elapsed > self.memory.frame_seconds:
                        frame = ImageHelper.resize_with_ratio(frame, self.memory.capture_width, None)

                        pub.sendMessage(
                            Event_Type.BRAIN_PROCESS,
                            frame=frame,
                            vision_date=vision_date,
                        )

                    if self.memory.source == Agent_Source.VISION and self.memory.pause_capture_seconds:
                        sleep(self.memory.pause_capture_seconds)

                if (self.memory.show_stream
                        and self.memory.perception is not None
                        and self.memory.perception.frame_with_detections is not None):
                    cv2.imshow('Perception', self.memory.perception.frame_with_detections)

                if self.memory.show_stream and cv2.waitKey(1) == ord('q'):
                    self.memory.disable('cv2')

        elif self.memory.source == Agent_Source.PHOTO:
            path = self.memory.get_last_memory()
            self.send_log('watch', path)

            frame = cv2.imread(path)

            pub.sendMessage(
                Event_Type.BRAIN_PROCESS,
                frame=frame,
                vision_date=vision_date,
            )

            if (self.memory.show_stream
                    and self.memory.perception is not None
                    and self.memory.perception.frame_with_detections is not None):
                cv2.imshow('Perception', self.memory.perception.frame_with_detections)

        if self.memory.brain_enabled and path is not None and os.path.isfile(path):
            # self.send_log('close', path)
            os.remove(path)

        if self.memory.show_stream:
            cv2.destroyAllWindows()
