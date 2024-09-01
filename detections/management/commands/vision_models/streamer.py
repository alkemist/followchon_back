import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import cv2
from loguru import logger
from vcgencmd import Vcgencmd

from configuration.models import Parameter, Log
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

        self.stop = False
        self.is_recording = False
        self.last_frame_seconds = 0
        self.last_capture_seconds = time.time()

        self.model = model

        self.vcgm = Vcgencmd()

        self.params_dict: dict[int, Parameter] = {}

    def log(self, event, info, level='info'):
        temp = self.vcgm.measure_temp()
        message = f"{event} : {info} temp={temp}°"

        match level:
            case 'info':
                logger.info(message)
            case 'warning':
                logger.warning(message)
            case 'error':
                logger.error(message)

        Log().create(self.model.current_model_version, 'vision', level, event, info, temp)

    def begin_recording(self, records_count=None) -> object | None:
        if records_count is not None:
            info = f"records={records_count}/{self.model.max_records} "
            f"hour={datetime.now().hour}h/{self.model.min_hour}h-{self.model.max_hour}h "

            self.log(
                'Start recording',
                info
            )

        command = (f"ffmpeg -hide_banner -y -loglevel error -rtsp_transport tcp -use_wallclock_as_timestamps "
                   f"1 -i {self.stream_path} -vcodec copy -acodec copy -f segment -reset_timestamps 1 "
                   f"-segment_time {self.record_time} -segment_format mkv -segment_atclocktime 1 -strftime 1 "
                   f"{self.records_directory}/%Y-%m-%d_%H-%M-%S.mkv")

        self.is_recording = True
        self.last_capture_seconds = time.time()

        return subprocess.Popen(command.split(" "),
                                stdout=subprocess.PIPE,
                                universal_newlines=True)

    def stop_recording(self):
        command = "pkill ffmpeg"

        self.is_recording = False

        return subprocess.Popen(command.split(" "),
                                stdout=subprocess.PIPE,
                                universal_newlines=True)

    def is_hour_ok(self):
        return self.model.min_hour <= datetime.now().hour <= self.model.max_hour

    def is_recording_ok(self, records_count):
        return (
                self.is_hour_ok() and
                (self.is_recording and records_count <= self.model.max_records or
                 not self.is_recording and records_count <= self.min_records_recording)
        )

    def start(self):
        self.log(
            'Started',
            f"records={self.get_records_count()}/{self.model.max_records} "
        )
        records_count = 0

        while not self.stop:
            records = FileHelper.list_files(self.records_directory, r'.*\.(mkv)$')
            records_count = len(records)

            if self.vcgm.measure_temp() > self.model.alert_temp:
                self.log(
                    'Temperature',
                    f"records={records_count}/{self.model.max_records} "
                    f"frame_seconds={self.model.frame_seconds}s "
                    f"fps={self.model.fps} "
                    f"pause={self.model.pause_minutes}m ",
                    "warning"
                )

            if self.model.loop_enabled and not self.is_recording and self.is_recording_ok(records_count):
                self.begin_recording(records_count)

            if self.is_recording and records_count >= self.model.max_records:
                self.log(
                    'Stop recording',
                    f"records={records_count}/{self.model.max_records} "
                )

                self.stop_recording()

            if records_count > self.min_records_capture or (
                    not self.model.loop_enabled
                    or not self.is_recording
            ) and records_count > 0:
                last_record = records[0]
                camera_record_filename = f"{self.records_directory}/{last_record}"

                self.capture(camera_record_filename)

                self.last_capture_seconds = time.time()

                if not self.stop and os.path.isfile(camera_record_filename):
                    os.remove(camera_record_filename)

            elif not self.model.loop_enabled and records_count <= 1:
                logger.info(
                    f"Vision finished : "
                    f"records={records_count}/{self.model.max_records} "
                    f"temp={self.vcgm.measure_temp()}° "
                )
                self.stop = True

            # Pas assez de vidéo ou temp trop chaud, on peut attendre un peu
            if (self.model.loop_enabled and (
                    records_count <= self.min_records_capture or self.vcgm.measure_temp() > self.model.max_temp
            ) and self.is_recording and not datetime.now().hour >= self.model.max_hour):

                self.model.release()

                time.sleep(self.model.pause_minutes * 60)

                self.last_capture_seconds = time.time()
                records_count = self.get_records_count()

                if self.is_recording_ok() and records_count < self.min_records_recording:
                    self.log(
                        'Restart recording',
                        f"records={records_count}/{self.model.max_records} "
                        f"frame_seconds={self.model.frame_seconds}s "
                        f"fps={self.model.fps} "
                        f"pause={self.model.pause_minutes}m "
                        f"recording={self.is_recording} "
                        ,
                        'warning'
                    )
                    self.begin_recording()

            if self.model.stop or datetime.now().hour > self.model.max_hour:
                self.log(
                    'Stopped',
                    f"records={records_count}/{self.model.max_records} "
                    f"stop={self.model.stop} "
                    f"hour={datetime.now().hour}h/{self.model.min_hour}h-{self.model.max_hour}h "
                )
                self.stop = True
            else:
                self.model.check_model()

        if self.show_stream:
            cv2.destroyAllWindows()

        logger.info(
            f"Finish recording : "
            f"records={records_count}/{self.model.max_records} "
            f"temp={self.vcgm.measure_temp()}° "
        )
        self.stop_recording()

        self.model.release()

    def capture(self, camera_record_filename: str):
        self.last_frame_seconds = 0
        file_date = Path(camera_record_filename).stem
        frame_count = 0

        cap = cv2.VideoCapture(camera_record_filename)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)
        cap.set(cv2.CAP_PROP_FPS, self.model.fps)

        while cap.isOpened() and not self.stop:
            frame_seconds_elapsed = time.time() - self.last_frame_seconds
            ret, frame = cap.read()

            if ret:
                # Si l'appareil n'est pas assez rapide pour analyser toutes les images
                if frame_seconds_elapsed > self.model.frame_seconds:
                    saved = self.infer(frame, frame_count, file_date)
                    self.last_frame_seconds = time.time()

                    if saved:
                        frame_count = frame_count + 1
            else:
                break

            if self.show_stream and cv2.waitKey(1) == ord('q'):
                self.stop = True

    def infer(self, frame: cv2.typing.MatLike, frame_count, datestr):
        (frame, saved) = self.model.infer(
            frame,
            frame_count,
            datestr
        )

        self.model.fill_params()
        self.stop = self.model.stop

        self.last_frame_seconds = time.time()

        if self.show_stream:
            cv2.imshow('Camera', frame)

        return saved

    def get_records_count(self):
        return len(FileHelper.list_files(self.records_directory, r'.*\.(mkv)$'))
