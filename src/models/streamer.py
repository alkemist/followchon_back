import os
import subprocess
import time

import cv2

from helpers.file import FileHelper
from .model import Model


class Streamer:

    def __init__(
            self,
            stream_path,
            model_path,
            records_directory,
            capture_width,
            capture_height,
            frame_time_seconds,
            capture_min_conf,
            check_all_records=False,
            delete_record=False,
            loop_enabled=False,
            save_enabled=False,
            track_enabled=False,
            verbose=False,
            show_stream=False,
    ):
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.check_all_records = check_all_records
        self.delete_record = delete_record
        self.loop_enabled = loop_enabled
        self.save_enabled = save_enabled
        self.track_enabled = track_enabled
        self.verbose = verbose
        self.show_stram = show_stream

        self.stop = False
        self.stream_path = stream_path
        self.records_directory = records_directory
        self.frame_time_seconds = frame_time_seconds
        self.capture_min_conf = capture_min_conf

        self.model = Model(model_path, capture_width, capture_height, capture_min_conf)

    def record(self):
        command = (f"ffmpeg -hide_banner -y -loglevel error -rtsp_transport tcp -use_wallclock_as_timestamps "
                   f"1 -i {self.stream_path} -vcodec copy -acodec copy -f segment -reset_timestamps 1 "
                   f"-segment_time 60 -segment_format mkv -segment_atclocktime 1 -strftime 1 "
                   f"{self.records_directory}/%Y-%m-%d_%H-%M-%S.mkv")

        return subprocess.Popen(command.split(" "),
                                stdout=subprocess.PIPE,
                                universal_newlines=True)

    def start(self):
        process = self.record()

        last_record_index = 0
        records = FileHelper.list_files(self.records_directory, r'.*\.(mkv)$')
        records_count = len(records)
        capture_time = time.time()

        while not self.stop:
            capture_time_elapsed = time.time() - capture_time

            if capture_time_elapsed >= 70:
                if self.verbose:
                    print('Restart recording')
                process = self.record()
                capture_time = time.time()

            if self.delete_record:
                records = FileHelper.list_files(self.records_directory, r'.*\.(mkv)$')
                records_count = len(records)

                last_record_index = records_count - 2 \
                    if records_count > 2 \
                    else 0

            if (records_count > 1 or not self.loop_enabled) and last_record_index < records_count:
                last_record = records[last_record_index]

                if self.verbose:
                    print(f"Next record : {last_record}")

                self.capture(last_record)
                capture_time = time.time()

            elif not self.loop_enabled:
                self.stop = True

            if (not self.loop_enabled and
                    (not self.delete_record or self.check_all_records)):
                last_record_index = last_record_index + 1

        if self.show_stram:
            cv2.destroyAllWindows()

        process.terminate()

    def capture(self, last_record):
        camera_record_filename = f"{self.records_directory}/{last_record}"
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
                        self.save_enabled,
                        self.verbose
                    )
                    frame_time = time.time()

                    if self.show_stram:
                        cv2.imshow('Camera', frame)
            else:
                break

            if self.show_stram and cv2.waitKey(1) == ord('q'):
                self.stop = True

        if self.verbose:
            print(f"End record : {last_record}")

        if self.delete_record:
            os.remove(camera_record_filename)
