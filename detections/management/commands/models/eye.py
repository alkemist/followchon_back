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
from detections.management.commands.models.enums.log_level import Log_Level
from detections.management.commands.models.memory import Memory
from utils.date import DateHelper
from utils.image import ImageHelper


class Eye:
    def __init__(
            self,
            memory: Memory,
    ):
        self.memory = memory

        self.last_frame_seconds = 0

    def send_log(self, event: str, infos: str = '', level: Log_Level = None):
        pub.sendMessage(Event_Type.AGENT_LOG, source=Event_Source.EYE, event=event, infos=infos, level=level)

    def watch(self):
        vision_date = datetime.now()
        path = None

        if self.memory.source == Agent_Source.VISION or self.memory.source == Agent_Source.VIDEO:
            path = self.memory.get_last_memory()

            self.memory.frame_count = 0
            self.memory.frame_saved_count = 0
            self.memory.last_record_seconds = time.time()
            self.memory.eye_start = time.time()

            if self.memory.source == Agent_Source.VISION:
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
            else:
                self.memory.last_detections = {}

            cap = cv2.VideoCapture(path)
            frames = 0
            duration = 0

            try:
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                duration = round(frames / fps, 1)
            except Exception as ex:
                fps = 0

            log = os.path.basename(path)

            if fps > 0 and frames > 0 and duration > 0:
                log = f"{log} / frames: {frames} / fps: {round(fps, 2)} / duration: {DateHelper.secondsToMMSS(duration)}"

            self.send_log('open', log)

            ret = True
            frames = 0

            while ret and cap.isOpened() and self.memory.brain_enabled:
                frame_seconds_elapsed = time.time() - self.last_frame_seconds
                ret, frame = cap.read()
                frames = frames + 1

                if ret and frame is not None and frame.size > 0:

                    if frame_seconds_elapsed > self.memory.frame_seconds:
                        self.memory.frame_count = self.memory.frame_count + 1

                        frame = ImageHelper.resize_with_ratio(frame, self.memory.capture_width, None)

                        pub.sendMessage(
                            Event_Type.BRAIN_PROCESS,
                            frame=frame,
                            vision_date=vision_date,
                        )

                        self.last_frame_seconds = time.time()

                        if self.memory.pause_capture_seconds > 0:
                            sleep(self.memory.pause_capture_seconds)

                if (self.memory.show_stream
                        and self.memory.perception is not None
                        and self.memory.perception.frame_with_detections is not None):
                    cv2.imshow('Perception', self.memory.perception.frame_with_detections)

                if self.memory.show_stream and cv2.waitKey(1) == ord('q'):
                    self.memory.terminate('cv2')

            if self.memory.brain_enabled:
                self.memory.add_statistics(frames, duration)

                if self.memory.frame_saved_count > self.memory.popcorn_frame_count and self.memory.source == Agent_Source.VISION:
                    self.memory.log_popcorn(vision_date)

            self.memory.add_temperature(True)

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
            os.remove(path)

        if self.memory.show_stream:
            cv2.destroyAllWindows()
