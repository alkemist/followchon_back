import os
import time

import cv2
from ultralytics import YOLO

from configuration.models import Family, Zone
from detections.management.commands.vision_models.capture_analyse import Capture_analyse
from detections.models import Detection
from helpers.array import ArrayHelper
from helpers.image import ImageHelper


class Model:

    def __init__(self, model_path: str, capture_min_score: float, capture_width: int, capture_height: int,
                 verbose: bool):
        self.model = YOLO(model_path, task='detect')

        self.capture_min_score = capture_min_score
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.verbose = verbose

        self.save_time = 0

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

    def detect(self, frame: cv2.typing.MatLike):
        results = self.model(frame, stream=True, verbose=False)

        save_time_elapsed = time.time() - self.save_time

        analyse = Capture_analyse(frame, self.last_detections_dict, self.families_dict, self.zones,
                                  self.capture_min_score)
        frame = analyse.detect(results)

        if analyse.is_triggered and save_time_elapsed > 1:
            analyse.save()
            self.save_time = time.time()

        return ImageHelper.resize_with_ratio(frame, int(os.getenv('CAPTURE_WIDTH')), None)
