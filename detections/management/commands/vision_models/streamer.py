import os
import subprocess
import time

import cv2

from detections.management.commands.vision_models.model import Model
from helpers.file import FileHelper


class Streamer:

    def __init__(self):
        self.stream_path = os.getenv('LIVE_STREAM_PATH')

        self.verbose = os.getenv('VERBOSE') == 'True'
        self.loop_enabled = os.getenv('LOOP_ENABLED') == 'True'
        self.show_stream = os.getenv('SHOW_STREAM') == 'True'
        self.frame_time_seconds = float(os.getenv('FRAME_TIME_SECONDS'))  # 0.03 < > 0.02
        self.capture_min_score = float(os.getenv('CAPTURE_MIN_SCORE'))  # 0.03 < > 0.02

        self.records_directory = './records'
        self.record_time = 60
        self.record_time_delay = 10
        self.capture_width = 1024
        self.capture_height = 768

        self.model = Model(
            os.getenv('MODEL_PATH'),
            self.capture_min_score,
            self.capture_width,
            self.capture_height,
            self.verbose
        )

        self.stop = False

    def record(self):
        command = (f"ffmpeg -hide_banner -y -loglevel error -rtsp_transport tcp -use_wallclock_as_timestamps "
                   f"1 -i {self.stream_path} -vcodec copy -acodec copy -f segment -reset_timestamps 1 "
                   f"-segment_time {self.record_time} -segment_format mkv -segment_atclocktime 1 -strftime 1 "
                   f"{self.records_directory}/%Y-%m-%d_%H-%M-%S.mkv")

        return subprocess.Popen(command.split(" "),
                                stdout=subprocess.PIPE,
                                universal_newlines=True)

    def start(self):
        process = self.record() if self.loop_enabled else None

        capture_time = time.time()

        while not self.stop:
            records = FileHelper.list_files(self.records_directory, r'.*\.(mkv)$')
            records_count = len(records)

            capture_time_elapsed = time.time() - capture_time

            if self.loop_enabled and capture_time_elapsed >= self.record_time + self.record_time_delay:
                if self.verbose:
                    print('Restart recording')
                process = self.record()
                capture_time = time.time()

            if records_count > 1 or not self.loop_enabled:
                last_record = records[0]
                camera_record_filename = f"{self.records_directory}/{last_record}"

                if self.verbose:
                    print(f"Next record : {last_record}")

                self.capture(camera_record_filename)

                capture_time = time.time()
                os.remove(camera_record_filename)

                if self.verbose:
                    print(f"End record : {last_record}")

            elif not self.loop_enabled and records_count <= 1:
                self.stop = True

        if self.show_stream:
            cv2.destroyAllWindows()

        if process is not None:
            process.terminate()

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
                    frame = self.model.detect(
                        frame,
                    )
                    frame_time = time.time()

                    if self.show_stream:
                        cv2.imshow('Camera', frame)
            else:
                break

            if self.show_stream and cv2.waitKey(1) == ord('q'):
                self.stop = True
