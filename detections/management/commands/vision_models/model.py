import os

import cv2

from configuration.models import Family, Zone
from detections.models import Detection
from helpers.array import ArrayHelper


class Model:

    def __init__(self):
        self.capture_min_score = float(os.getenv('CAPTURE_MIN_SCORE'))  # 0.03 < > 0.02
        self.verbose = os.getenv('VERBOSE') == 'True'

        self.capture_min_score = float(os.getenv('CAPTURE_MIN_SCORE'))  # 0.03 < > 0.02
        self.verbose = os.getenv('VERBOSE') == 'True'

        self.save_time = 0

        self.families = Family.objects.all()
        self.zones = Zone.objects.all()
        self.capture_width = int(os.getenv('CAPTURE_WIDTH'))
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

        self.stop = False

    def infer(self, frame: cv2.typing.MatLike):
        raise Exception('Infer not implemented')
