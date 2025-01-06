import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from time import sleep

import cv2

from detections.management.commands.vision_models.source import Source
from detections.management.commands.vision_models.supervisor import Supervisor
from detections.management.commands.vision_models.vision import Vision


class Streamer:

    def __init__(self, supervisor: Supervisor):
        self.supervisor = supervisor
        self.vision = Vision(supervisor)

        self.stream_path = os.getenv('LIVE_STREAM_PATH')

        self.show_stream = os.getenv('SHOW_STREAM') == 'True'

    def begin_recording(self, log: bool = False) -> object | None:
        if log:
            self.supervisor.log_start_recording()

        command = (f"ffmpeg -hide_banner -y -loglevel error -rtsp_transport tcp -use_wallclock_as_timestamps "
                   f"1 -i {self.stream_path} -vcodec copy -acodec copy -f segment -reset_timestamps 1 "
                   f"-segment_time {self.supervisor.record_time} -segment_format mkv -segment_atclocktime 1 -strftime 1 "
                   f"{self.supervisor.records_directory}/%Y-%m-%d_%H-%M-%S.mkv")

        self.supervisor.begin_recording()

        return subprocess.Popen(command.split(" "),
                                stdout=subprocess.PIPE,
                                universal_newlines=True)

    def stop_recording(self):
        command = "pkill ffmpeg"

        self.supervisor.stop_recording()

        return subprocess.Popen(command.split(" "),
                                stdout=subprocess.PIPE,
                                universal_newlines=True)

    def long_sleep(self, time_minutes):
        self.supervisor.local_log(f'Long sleep start')

        self.vision.release()
        time.sleep(time_minutes * 60)
        self.supervisor.local_log(f'Long sleep end')
        self.supervisor.last_capture_seconds = time.time()

    def start(self):
        self.vision.check_model('start')
        self.supervisor.log_start()

        hour = datetime.now().hour

        while self.supervisor.enabled:
            self.supervisor.fill_params()
            self.supervisor.add_temperature()

            records = self.supervisor.get_records()
            capture_time_elapsed = round(time.time() - self.supervisor.last_capture_seconds)

            if (datetime.now().hour > hour
                    and self.supervisor.hour_min < datetime.now().hour < self.supervisor.hour_max):
                hour = datetime.now().hour
                self.supervisor.log_hourly()

            if (not self.supervisor.is_recording
                    and self.supervisor.is_recording_ok()
            ):
                self.begin_recording(True)

            if (self.supervisor.is_recording_ok()
                    and capture_time_elapsed >= self.supervisor.record_time + self.supervisor.record_time_delay
            ):
                self.supervisor.log_restart_recording(capture_time_elapsed)
                self.begin_recording()

            if self.supervisor.is_recording and (self.supervisor.records_count >= self.supervisor.records_max
                                                 or datetime.now().hour >= self.supervisor.hour_max):
                self.supervisor.log_stop_recording()
                self.stop_recording()

            if (self.supervisor.records_count > self.supervisor.min_records_capture \
                    or (datetime.now().hour >= self.supervisor.hour_max or self.supervisor.source != Source.VISION)
                    and self.supervisor.records_count > 0):
                last_record = records[0]
                camera_record_filename = f"{self.supervisor.records_directory}/{last_record}"

                # self.supervisor.local_log('Begin file', camera_record_filename)

                self.supervisor.delay_start = time.time()
                duration = self.capture(camera_record_filename)

                self.supervisor.last_capture_seconds = time.time()

                if self.supervisor.enabled and os.path.isfile(camera_record_filename):
                    if self.supervisor.source == Source.VISION:
                        self.supervisor.add_processing_delay()
                    else:
                        self.vision.last_detections_dict = {}

                    # self.supervisor.local_log('End file', camera_record_filename)
                    os.remove(camera_record_filename)

            # Pas assez de vidéo, on peut attendre un peu
            if (self.supervisor.records_count <= self.supervisor.min_records_capture
                    and self.supervisor.pause_records_minutes
                    and self.supervisor.is_recording
                    and datetime.now().hour < self.supervisor.hour_max
            ):
                records_count_before = self.supervisor.records_count

                # self.supervisor.log_sleeping()
                # self.supervisor.add_temperature(True)

                self.long_sleep(self.supervisor.pause_records_minutes)

                # self.supervisor.add_temperature(True)
                self.supervisor.get_records_count()

                # self.supervisor.log_awakened()

                capture_minutes = self.supervisor.record_time / 60
                capture_count_new = (self.supervisor.pause_records_minutes / capture_minutes)
                capture_count_margin = capture_count_new / 10 \
                    if self.supervisor.pause_records_minutes > 10 \
                    else 1 if self.supervisor.pause_records_minutes > 1 else 0

                if (self.supervisor.is_recording_ok()
                        and self.supervisor.records_count <
                        records_count_before + capture_count_new - capture_count_margin
                ):
                    self.supervisor.log_restart_recording()
                    self.begin_recording()

            if self.supervisor.temp > self.supervisor.temp_alert:
                self.supervisor.log_warning_temperature()
                self.supervisor.add_temperature(True)

                self.long_sleep(self.supervisor.temp_pause)

                self.supervisor.add_temperature(True)

            if not self.supervisor.is_recording and datetime.now().hour < self.supervisor.hour_min:
                # self.supervisor.log_waiting()
                self.long_sleep(60)

            if ((datetime.now().hour >= self.supervisor.hour_max or self.supervisor.source != Source.VISION)
                    and self.supervisor.records_count == 0
            ):
                self.supervisor.enabled = False

        if self.show_stream:
            cv2.destroyAllWindows()

        self.supervisor.log_stopped()
        self.supervisor.log_stat_processing()
        self.supervisor.log_stat_fpm()

        if self.supervisor.vcgm is not None:
            self.supervisor.log_stat_temperature()

        if self.supervisor.source == Source.VISION:
            self.stop_recording()

        self.vision.release()

    def capture(self, camera_record_filename: str):
        self.supervisor.last_frame_seconds = 0

        if self.supervisor.source == Source.VISION:
            file_date = Path(camera_record_filename).stem
            date_values = re.split('[-_]', file_date)
            capture_date = datetime(
                int(date_values[0]),
                int(date_values[1]),
                int(date_values[2]),
                int(date_values[3]),
                int(date_values[4]),
                int(date_values[5]),
            )
        else:
            capture_date = datetime.now()

        frame_saved_count = 0
        analyse_count = 0

        cap = cv2.VideoCapture(camera_record_filename)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)

        while cap.isOpened() and self.supervisor.enabled:

            frame_seconds_elapsed = time.time() - self.supervisor.last_frame_seconds
            ret, frame = cap.read()

            # self.supervisor.local_log('Read frame', str(ret))

            if ret:
                # Si l'appareil n'est pas assez rapide pour analyser toutes les images
                if frame_seconds_elapsed > self.supervisor.frame_seconds:
                    # self.supervisor.local_log('Infer frame')

                    saved = self.infer(frame, frame_saved_count, capture_date)
                    self.supervisor.last_frame_seconds = time.time()

                    analyse_count = analyse_count + 1
                    if saved:
                        frame_saved_count = frame_saved_count + 1

                    if self.supervisor.pause_capture_seconds:
                        # self.supervisor.local_log('Pause', str(self.supervisor.pause_capture_seconds))
                        sleep(self.supervisor.pause_capture_seconds)
            else:
                break

            if self.show_stream and cv2.waitKey(1) == ord('q'):
                self.supervisor.enabled = False

            # self.supervisor.local_log('End frame', str(cap.isOpened()))

        if self.supervisor.source == Source.VISION:
            if frame_saved_count > self.supervisor.popcorn_frame_count:
                self.supervisor.log_popcorn(capture_date, frame_saved_count)

            if self.supervisor.enabled:
                self.supervisor.analyses_by_record.append(analyse_count)

        if fps is not None and fps > 0:
            return round(frames / fps)

        return None

    def read(self):
        self.vision.check_model('read')
        self.supervisor.log_start()

        frame_saved_count = 0

        while self.supervisor.enabled:
            self.supervisor.fill_params()

            records = self.supervisor.get_records()

            if self.supervisor.records_count > 0:
                last_record = records[0]
                photo_filename = f"{self.supervisor.records_directory}/{last_record}"

                frame = cv2.imread(photo_filename)

                saved = self.infer(frame, frame_saved_count, datetime.now())

                if saved:
                    frame_saved_count = frame_saved_count + 1

                if self.supervisor.pause_capture_seconds:
                    sleep(self.supervisor.pause_capture_seconds)

                if self.supervisor.enabled and os.path.isfile(photo_filename):
                    os.remove(photo_filename)

            else:
                self.supervisor.enabled = False

            if self.show_stream and cv2.waitKey(1) == ord('q'):
                self.supervisor.enabled = False

        if self.show_stream:
            cv2.destroyAllWindows()

        self.supervisor.log_stopped()

        self.vision.release()

    def infer(self, frame: cv2.typing.MatLike, frame_count, capture_date):
        (frame, saved) = self.vision.infer(
            frame, frame_count, capture_date
        )

        self.supervisor.last_frame_seconds = time.time()

        if self.show_stream:
            cv2.imshow('Camera', frame)

        return saved

    def release(self):
        self.vision.release()
