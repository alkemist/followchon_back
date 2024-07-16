import time

from ultralytics import YOLO

from configuration.models import Family, Zone
from detections.management.commands.vision_models.analyse import Analyse
from detections.models import Detection
from helpers.array import ArrayHelper
from helpers.image import ImageHelper


class Model:

    def __init__(self, model_path, capture_min_score, capture_width, capture_height, verbose):
        self.model = YOLO(model_path, task='detect')

        self.capture_min_score = capture_min_score
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.verbose = verbose

        self.save_time = 0

        self.families = Family.objects.all()
        self.zones = Zone.objects.all()
        self.families_dict = ArrayHelper.object_list_to_dict(self.families, 'index')
        self.families_trigger = [family for family in self.families if family.trigger]
        self.families_trigger_count = {}

        for family_trigger in self.families_trigger:
            self.families_trigger_count[family_trigger.index] = 1

            if family_trigger.parent:
                if family_trigger.parent.index not in self.families_trigger_count:
                    self.families_trigger_count[family_trigger.parent.index] = 0
                self.families_trigger_count[family_trigger.parent.index] += 1

        last_detections = Detection.objects.raw(
            'SELECT * FROM' +
            ' (SELECT * FROM detections_detection d' +
            ' LEFT JOIN configuration_family f ON d.family_id = f.id' +
            ' WHERE f.tracked = true'
            ' ORDER BY d.id DESC) d' +
            ' GROUP BY d.family_id')

        self.last_detections_dict = ArrayHelper.object_list_to_dict(last_detections, 'family_id')

    def detect(self, frame):
        results = self.model(frame, stream=True, verbose=False)

        save_time_elapsed = time.time() - self.save_time

        analyse = Analyse(frame,
                          self.families_dict, self.families_trigger_count, self.zones,
                          self.capture_min_score, self.capture_width,
                          self.capture_height)
        frame = analyse.detect(results)

        return ImageHelper.resize_with_ratio(frame, self.capture_width, self.capture_height)
