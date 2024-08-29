import os
import subprocess
import threading
import time
from datetime import datetime

import cv2
from loguru import logger
from vcgencmd import Vcgencmd

from configuration.models import Parameter
from detections.management.commands.vision_models.model import Model
from helpers.file import FileHelper


class Streamer:

    def __init__(self, model: Model):
        self.stream_path = os.getenv('LIVE_STREAM_PATH')

        self.show_stream = os.getenv('SHOW_STREAM') == 'True'

        self.records_directory = './records'
        self.record_time = 60
        self.record_time_delay = 50
        self.min_records_capture = 1
        self.min_records_recording = 2
        self.capture_width = 1024
        self.capture_height = 768
        self.temp = 0

        self.stop = False
        self.is_recording = False
        self.last_frame_seconds = 0

        self.model = model

        self.vcgm = Vcgencmd()

        self.params_dict: dict[int, Parameter] = {}

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
        return (
                self.model.min_hour <= datetime.now().hour <= self.model.max_hour and
                (self.is_recording and records_count <= self.model.max_records or
                 not self.is_recording and records_count <= self.min_records_recording)
        )

    def start(self):
        self.temp = self.vcgm.measure_temp()

        records = FileHelper.list_files(self.records_directory, r'.*\.(mkv)$')
        records_count = len(records)
        capture_time = time.time()

        logger.info(
            f"Vision started : "
            f"count={records_count}/{self.model.max_records} "
            f"temp={self.temp}° "
        )

        if self.model.loop_enabled and self.is_recording_ok(records_count):
            logger.info(
                f"Start recording : "
                f"hour={datetime.now().hour}h/{self.model.min_hour}h-{self.model.max_hour}h "
                f"count={records_count}/{self.model.max_records} "
                f"temp={self.temp}° "
            )
            self.begin_recording()

        while not self.stop:
            self.temp = self.vcgm.measure_temp()

            records = FileHelper.list_files(self.records_directory, r'.*\.(mkv)$')
            records_count = len(records)

            capture_time_elapsed = round(time.time() - capture_time, 2)

            if (
                    self.model.loop_enabled
                    and capture_time_elapsed >= self.record_time_delay
                    or not self.is_recording
            ) and self.is_recording_ok(records_count):
                logger.info(
                    f"Start recording : "
                    f"time={capture_time_elapsed}s/{self.record_time_delay}s "
                    f"hour={datetime.now().hour}h/{self.model.min_hour}h-{self.model.max_hour}h "
                    f"count={records_count}/{self.model.max_records} "
                    f"temp={self.temp}° "
                )

                self.begin_recording()
                capture_time = time.time()

            if self.is_recording and records_count >= self.model.max_records:
                logger.info(
                    f"Stop recording : "
                    f"count={records_count}/{self.model.max_records} "
                    f"temp={self.temp}° "
                )

                self.stop_recording()

            if records_count > self.min_records_capture or (
                    not self.model.loop_enabled
                    or not self.is_recording
            ) and records_count > 0:
                last_record = records[0]
                camera_record_filename = f"{self.records_directory}/{last_record}"

                self.capture(camera_record_filename)

                capture_time = time.time()

                if not self.stop and os.path.isfile(camera_record_filename):
                    os.remove(camera_record_filename)

            elif not self.model.loop_enabled and records_count <= 1:
                logger.info(
                    f"Vision finished : "
                    f"count={records_count}/{self.model.max_records} "
                    f"temp={self.temp}° "
                )
                self.stop = True

            # Pas assez de vidéo ou temp trop chaud, on peut attendre un peu
            if self.model.loop_enabled and (records_count <= self.min_records_capture or self.temp > 80):
                logger.info(
                    f"Vision sleep : "
                    f"count={records_count}/{self.model.max_records} "
                    f"frame_seconds={self.model.frame_seconds}s "
                    f"pause={self.model.pause_minutes}m "
                    f"temp={self.temp}° "
                )

                self.model.release()

                threading.Event().wait(self.model.pause_minutes * 60)

                records_count = len(FileHelper.list_files(self.records_directory, r'.*\.(mkv)$'))
                capture_time = time.time()

                logger.info(
                    f"Vision awake : "
                    f"count={records_count}/{self.model.max_records} "
                    f"frame_seconds={self.model.frame_seconds}s "
                    f"pause={self.model.pause_minutes}m "
                    f"temp={self.temp}° "
                )

            if self.model.stop or datetime.now().hour > self.model.max_hour:
                logger.info(
                    f"Vision stopped : "
                    f"stop={self.model.stop} "
                    f"hour={datetime.now().hour}h/{self.model.min_hour}h-{self.model.max_hour}h "
                    f"temp={self.temp}° "
                )
                self.stop = True
            else:
                self.model.check_model()

        if self.show_stream:
            cv2.destroyAllWindows()

        if self.is_recording:
            logger.info(
                f"Finish recording : "
                f"count={records_count}/{self.model.max_records} "
                f"temp={self.temp}° "
            )
            self.stop_recording()

        self.model.release()

    def capture(self, camera_record_filename: str):
        self.last_frame_seconds = 0

        cap = cv2.VideoCapture(camera_record_filename)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)

        while cap.isOpened() and not self.stop:
            self.temp = self.vcgm.measure_temp()

            frame_seconds_elapsed = time.time() - self.last_frame_seconds
            ret, frame = cap.read()

            if ret:
                # Si l'appareil n'est pas assez rapide pour analyser toutes les images
                if frame_seconds_elapsed > self.model.frame_seconds:
                    self.infer(frame)
            else:
                break

            if self.show_stream and cv2.waitKey(1) == ord('q'):
                self.stop = True

    def infer(self, frame: cv2.typing.MatLike):
        frame = self.model.infer(
            frame,
        )

        self.model.fill_params()
        self.stop = self.model.stop

        self.last_frame_seconds = time.time()

        if self.show_stream:
            cv2.imshow('Camera', frame)
