import os
import time

import cv2
from loguru import logger

from configuration.models import Family, Zone, Parameter
from detections.management.commands.vision_models.capture_analyse import Capture_analyse
from detections.models import Detection
from helpers.array import ArrayHelper


class Model:

    def __init__(self):
        self.model = None
        self.current_model_version = ''
        self.model_version = ''
        self.model_path = ''
        self.min_score = 0
        self.min_hour = 0
        self.max_hour = 0
        self.max_records = 0
        self.max_temp = 0
        self.alert_temp = 0
        self.pause_minutes = 0
        self.frame_seconds = 0
        self.fps = 0
        self.stop = False
        self.loop_enabled = False
        self.params_dict = {}

        self.save_time = 0
        self.capture_width = int(os.getenv('CAPTURE_WIDTH'))

        self.families = []
        self.zones = []
        self.families_dict = {}
        self.last_detections_dict = {}

    def get_params(self):
        parameters = Parameter.objects.all()
        self.params_dict: dict[int, Parameter] = (
            ArrayHelper.object_list_to_dict(parameters, 'slug')
        )

    def get_param(self, param):
        if param in self.params_dict:
            return self.params_dict[param].value

        logger.error(f'Param "{param}" not exist')
        return None

    def fill_objects(self):
        self.families = Family.objects.all()
        self.zones = Zone.objects.all()
        self.families_dict = ArrayHelper.object_list_to_dict(self.families, 'index')

        last_detections = Detection.objects.raw(
            'SELECT * FROM' +
            ' (SELECT * FROM detections_detection d' +
            ' LEFT JOIN configuration_family f ON d.family_id = f.id' +
            ' WHERE f.is_tracked = true'
            ' ORDER BY d.id DESC) d' +
            ' GROUP BY d.family_id')

        last_detections_dict: dict[int, Detection] = (
            ArrayHelper.object_list_to_dict(last_detections, 'family_id')
        )

        self.last_detections_dict: dict[int, Zone] = (
            dict(map(lambda kv: (kv[0], kv[1].zone), last_detections_dict.items())))

    def fill_params(self):
        self.get_params()

        self.model_version = int(self.get_param('vision_model_version'))
        self.min_score = float(self.get_param('vision_score_min'))
        self.min_hour = int(self.get_param('vision_hour_min'))
        self.max_hour = int(self.get_param('vision_hour_max'))
        self.max_records = int(self.get_param('vision_records_max'))
        self.pause_minutes = int(self.get_param('vision_pause_minutes'))
        self.max_temp = int(self.get_param('vision_temp_max'))
        self.alert_temp = int(self.get_param('vision_temp_alert'))
        self.frame_seconds = float(self.get_param('vision_frame_seconds'))  # 0.03 < > 0.02
        self.fps = int(self.get_param('vision_fps'))
        self.loop_enabled = self.get_param('vision_loop_enabled') == '1'
        self.stop = self.get_param('vision_stop') == '1'

    def analyze(self, frame, frame_count, datestr, yolo_results):
        saved = False

        if len(yolo_results) > 0:
            save_time_elapsed = time.time() - self.save_time

            analyse = Capture_analyse(
                frame, datestr, frame_count,
                self.last_detections_dict, self.families_dict, self.zones
            )

            frame = analyse.detect(yolo_results)

            if analyse.is_triggered and save_time_elapsed > 1:
                analyse.save()
                self.save_time = time.time()
                saved = True

        return frame, saved

    def get_model_path(self):
        return (f"{os.getenv('MODEL_DIR')}/"
                f"{os.getenv('MODEL_PREFIX')}{self.current_model_version}.{os.getenv('MODEL_EXT')}")

    def infer(self, frame_count, frame: cv2.typing.MatLike, datestr):
        raise Exception('Infer not implemented')

    def check_model(self):
        raise Exception('Check model not implemented')

    def release(self):
        return False
