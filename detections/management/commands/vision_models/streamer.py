import os
import subprocess
import time
from datetime import datetime

import cv2
from loguru import logger

from detections.management.commands.vision_models.model import Model
from helpers.file import FileHelper


class Streamer:

    def __init__(self, model: Model):
        self.stream_path = os.getenv('LIVE_STREAM_PATH')

        self.loop_enabled = os.getenv('LOOP_ENABLED') == 'True'
        self.show_stream = os.getenv('SHOW_STREAM') == 'True'
        self.frame_time_seconds = float(os.getenv('FRAME_TIME_SECONDS'))  # 0.03 < > 0.02
        self.verbose = os.getenv('VERBOSE') == 'True'
        self.min_hour = int(os.getenv('CAPTURE_MIN_HOUR'))
        self.max_hour = int(os.getenv('CAPTURE_MAX_HOUR'))

        self.records_directory = './records'
        self.record_time = 60
        self.record_time_delay = 10
        self.max_records = 10
        self.capture_width = 1024
        self.capture_height = 768

        self.stop = False
        self.is_recording = False

        self.model = model

        logger.add(f"{os.getenv('LOG_DIRECTORY')}streamer.log", rotation="1 days", retention=7)

    def log(self, message):
        if self.verbose:
            print(message)
        else:
            logger.info(message)

    def begin_recording(self) -> object | None:
        command = (f"ffmpeg -hide_banner -y -loglevel error -rtsp_transport tcp -use_wallclock_as_timestamps "
                   f"1 -i {self.stream_path} -vcodec copy -acodec copy -f segment -reset_timestamps 1 "
                   f"-segment_time {self.record_time} -segment_format mkv -segment_atclocktime 1 -strftime 1 "
                   f"{self.records_directory}/%Y-%m-%d_%H-%M-%S.mkv")

        self.is_recording = True

        return subprocess.Popen(command.split(" "),
                                stdout=subprocess.PIPE,
                                universal_newlines=True)

    def stop_recording(self):
        command = "pkill ffmpeg"

        self.is_recording = False

        return subprocess.Popen(command.split(" "),
                                stdout=subprocess.PIPE,
                                universal_newlines=True)

    def is_recording_ok(self, records_count):
        return self.min_hour <= datetime.now().hour <= self.max_hour and records_count <= self.max_records

    def start(self):
        records = FileHelper.list_files(self.records_directory, r'.*\.(mkv)$')
        records_count = len(records)
        capture_time = 0

        if self.loop_enabled and self.is_recording_ok(records_count):
            self.log(f"Start recording")
            self.begin_recording()
            capture_time = time.time()

        while not self.stop:
            records = FileHelper.list_files(self.records_directory, r'.*\.(mkv)$')
            records_count = len(records)

            capture_time_elapsed = time.time() - capture_time

            if (
                    self.loop_enabled and capture_time_elapsed >= self.record_time + self.record_time_delay
                    or not self.is_recording
            ) and self.is_recording_ok(records_count):
                self.log(
                    f"Start recording : time={capture_time_elapsed}/{self.record_time + self.record_time_delay} "
                    f"hour={datetime.now().hour}/{self.min_hour}-{self.max_hour}"
                )

                self.begin_recording()
                capture_time = time.time()

            if self.is_recording and not self.is_recording_ok(records_count):
                self.log(
                    f"Stop recording : count={records_count}/{self.max_records} "
                    f"hour={datetime.now().hour}/{self.min_hour}-{self.max_hour}"
                )

                self.stop_recording()

            if records_count > 1 or (not self.loop_enabled or not self.is_recording) and records_count > 0:
                last_record = records[0]
                camera_record_filename = f"{self.records_directory}/{last_record}"

                # self.log(f"Next record : {last_record}")

                self.capture(camera_record_filename)

                capture_time = time.time()

                if os.path.isfile(camera_record_filename):
                    os.remove(camera_record_filename)

                # self.log(f"End record : {last_record}")

            elif not self.loop_enabled and records_count <= 1:
                self.stop = True

        if self.show_stream:
            cv2.destroyAllWindows()

        if self.loop_enabled and self.is_recording:
            self.stop_recording()

    def capture(self, camera_record_filename: str):
        frame_time = 0

        cap = cv2.VideoCapture(camera_record_filename)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)

        while cap.isOpened() and not self.stop:
            frame_time_elapsed = time.time() - frame_time
            ret, frame = cap.read()

            if ret:
                if frame_time_elapsed > self.frame_time_seconds:
                    frame = self.model.infer(
                        frame,
                    )
                    self.stop = self.model.stop

                    frame_time = time.time()

                    if self.show_stream:
                        cv2.imshow('Camera', frame)
            else:
                break

            if self.show_stream and cv2.waitKey(1) == ord('q'):
                self.stop = True
