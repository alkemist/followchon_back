import os
import random
import re
import time
from datetime import datetime
from pathlib import Path
from time import sleep

import cv2
from pubsub import pub

from detections.management.commands.models.brain import Brain
from detections.management.commands.models.enums.agent_source import Agent_Source
from detections.management.commands.models.enums.event_source import Event_Source
from detections.management.commands.models.enums.event_type import Event_Type
from detections.management.commands.models.enums.log_level import Log_Level
from detections.management.commands.models.memory import Memory
from utils.date import DateHelper


class Eye:
    def __init__(
            self,
            memory: Memory,
            brain: Brain,
    ):
        self.memory = memory
        self.brain = brain

    def send_log(self, event: str, infos: str = '', level: Log_Level = None):
        pub.sendMessage(Event_Type.AGENT_LOG, source=Event_Source.EYE, event=event, infos=infos, level=level)

    def watch(self):
        vision_date = datetime.now()
        path = None

        if self.memory.source == Agent_Source.VISION or self.memory.source == Agent_Source.VIDEO:
            path = self.memory.get_last_memory()

            if path is None:
                return

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
            frames_total = 0
            duration = 0

            try:
                frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                duration = round(frames_total / fps, 1)
            except Exception as ex:
                fps = 0

            log = os.path.basename(path)

            if fps > 0 and frames_total > 0 and duration > 0:
                log = f"{log} / frames: {frames_total} / fps: {round(fps, 2)} / duration: {DateHelper.secondsToMMSS(duration)}"

            self.send_log('open', f"{log}")

            ret = True
            frames_total = 0
            frames_detected = 0
            frames_classified = 0
            frames_saved = 0
            frame_to_ignore = 0

            while ret and cap.isOpened() and self.memory.brain_enabled:
                ret, frame = cap.read()

                if ret and frame is not None and frame.size > 0:
                    frames_total = frames_total + 1

                    if frame_to_ignore <= 0:
                        frames_detected = frames_detected + 1

                        self.brain.process_neurons(
                            frame=frame,
                            vision_date=vision_date,
                            frame_count=frames_saved,
                        )

                        frame_to_ignore = self.memory.frames_detected_step

                        if not self.memory.perception.is_empty:
                            frames_classified = frames_classified + 1
                            frame_to_ignore = self.memory.frames_classified_step * self.memory.perception.detections_count

                        if self.memory.perception.is_saved:
                            frames_saved = frames_saved + 1
                            frame_to_ignore = self.memory.frames_saved_step

                        if self.memory.pause_capture_seconds > 0:
                            sleep(self.memory.pause_capture_seconds)
                    else:
                        frame_to_ignore = frame_to_ignore - 1

                if (self.memory.show_stream
                        and self.memory.perception is not None
                        and self.memory.perception.frame_with_detections is not None):
                    cv2.imshow('Perception', self.memory.perception.frame_with_detections)

                if self.memory.show_stream and cv2.waitKey(1) == ord('q'):
                    self.memory.terminate('cv2')

            if self.memory.brain_enabled:
                if duration > 0 and frames_total > 0:
                    duration_percent = int(round(
                        (time.time() - self.memory.eye_start) / duration,
                        2
                    ) * 100)

                    frames_detected_percent = int(round(frames_detected / frames_total, 2) * 100)
                    frames_classified_percent = int(round(frames_classified / frames_total, 2) * 100)
                    frames_saved_percent = int(round(frames_saved / frames_total, 2) * 100)

                    self.memory.durations.append(duration_percent)
                    self.memory.fpm_counts.append(frames_detected_percent)

                    self.send_log('close',
                                  f"duration: {duration_percent} / detected: {frames_detected_percent} / classified: {frames_classified_percent} / saved: {frames_saved_percent}" +
                                  f" with classify step: {self.memory.frames_classified_step} / save step: {self.memory.frames_saved_step} / pause: {self.memory.pause_capture_seconds}\n")

                if frames_saved > self.memory.popcorn_frame_count and self.memory.source == Agent_Source.VISION:
                    self.memory.log_popcorn(vision_date, frames_saved)

        elif self.memory.source == Agent_Source.PHOTO:
            path = self.memory.get_last_memory()
            self.send_log('watch', path)

            frame = cv2.imread(path)

            self.brain.process_neurons(
                frame=frame,
                vision_date=vision_date
            )

            if (self.memory.show_stream
                    and self.memory.perception is not None
                    and self.memory.perception.frame_with_detections is not None):
                cv2.imshow('Perception', self.memory.perception.frame_with_detections)

        if self.memory.brain_enabled and path is not None and os.path.isfile(path):
            os.remove(path)

        if self.memory.show_stream:
            cv2.destroyAllWindows()
