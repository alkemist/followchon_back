from detections.management.commands.vision_models.annotation import Annotation
from helpers.array import ArrayHelper


class Analyse:

    def __init__(self, frame, families_dict, families_trigger_count, zones, capture_min_score, capture_width,
                 capture_height):
        self.frame = frame

        self.capture_min_score = capture_min_score
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.families_dict = families_dict
        self.families_trigger_count = families_trigger_count
        self.zones = zones

        self.annotations = list()
        self.families_detect_count = {}

    def detect(self, results):
        frame_copy = self.frame.copy()

        for r in results:
            boxes = r.boxes

            for box in boxes:
                annotation = Annotation(box, self.capture_width, self.capture_height, self.families_dict, self.zones)

                if annotation.score > self.capture_min_score:
                    if annotation.family_index not in self.families_detect_count:
                        self.families_detect_count[annotation.family_index] = 0

                    self.families_detect_count[annotation.family_index] += 1
                    self.annotations.append(annotation)

        trigger_verified = len(self.families_trigger_count.keys()) == 0

        if not trigger_verified:
            for family_index in self.families_trigger_count.keys():
                trigger_verified = (
                        family_index in self.families_detect_count and
                        self.families_trigger_count[family_index] == self.families_detect_count[family_index]
                )

                if not trigger_verified:
                    break

        if trigger_verified:
            trigger_verified = (
                any([annotation.zone is not None and annotation.zone.trigger for annotation in self.annotations]))

        self.annotations = ArrayHelper.sort(self.annotations, lambda a1, a2: a1.family_index - a2.family_index)

        for annotation in self.annotations:
            frame_copy = annotation.trace(frame_copy)

        return frame_copy
