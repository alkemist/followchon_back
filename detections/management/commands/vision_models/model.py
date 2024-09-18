import os
import time

import cv2

from configuration.models import Family, Zone
from detections.management.commands.vision_models.capture_analyse import Capture_analyse
from detections.management.commands.vision_models.supervisor import Supervisor
from detections.models import Detection
from helpers.array import ArrayHelper


class Model:

    def __init__(self, supervisor: Supervisor):
        self.supervisor = supervisor
        self.model = None

        self.save_time = 0
        self.capture_width = int(os.getenv('CAPTURE_WIDTH'))

        self.families = []
        self.zones = []
        self.families_dict = {}
        self.last_detections_dict = {}

        self.families = Family.objects.all()
        self.families_dict = ArrayHelper.object_list_to_dict(self.families, 'index')

        last_detections = Detection.objects.raw(
            'SELECT * FROM (' +
            '    SELECT * FROM detections_detection d' +
            '    LEFT JOIN configuration_family f ON d.family_id = f.id' +
            '    WHERE f.is_tracked = true'
            '    ORDER BY d.id DESC'
            ' ) d' +
            ' GROUP BY d.family_id')

        last_detections_dict: dict[int, Detection] = (
            ArrayHelper.object_list_to_dict(last_detections, 'family_id')
        )

        self.last_detections_dict: dict[int, (float, float)] = (
            dict(
                map(
                    lambda kv: (kv[0], (kv[1].center_x, kv[1].center_y)),
                    last_detections_dict.items()
                )
            )
        )

    def fill_objects(self):
        self.zones = Zone.objects.all().filter(is_enabled=True).order_by('id')

    def analyze(self, frame, frame_count, capture_date, yolo_results):
        saved = False

        if len(yolo_results) > 0:
            analyse = Capture_analyse(
                frame, capture_date, frame_count,
                self.last_detections_dict, self.families_dict, self.zones,
                self.supervisor
            )

            frame = analyse.detect(yolo_results)

            if analyse.is_triggered:
                analyse.save()

                self.save_time = time.time()
                saved = True

        return frame, saved

    def infer(self, frame_count, frame: cv2.typing.MatLike, datestr):
        raise Exception('Infer not implemented')

    def check_model(self):
        raise Exception('Check model not implemented')

    def release(self):
        return False
