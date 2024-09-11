import copy
from datetime import datetime
from typing import cast

import cv2

from configuration.models import Family, Zone
from detections.management.commands.vision_models.annotation import Annotation
from detections.management.commands.vision_models.supervisor import Supervisor
from detections.models import Capture, Detection
from helpers.array import ArrayHelper


class Capture_analyse:

    def __init__(self,
                 frame: cv2.typing.MatLike, capture_date: datetime, frame_count,
                 last_detections_dict: dict[int, (float, float)],
                 families_dict: dict[int, Family], zones: list[Zone],
                 supervisor: Supervisor
                 ):
        self.frame = frame
        self.supervisor = supervisor

        self.date_capture = datetime(
            capture_date.year,
            capture_date.month,
            capture_date.day,
            capture_date.hour,
            capture_date.minute,
            capture_date.second,
            frame_count,
        )

        self.families_dict = families_dict
        self.last_detections_dict = last_detections_dict
        self.zones = zones

        self.families_detect_count = {}
        self.is_triggered = False
        self.annotations = list()

    def detect(self, results: list):
        frame_copy = self.frame.copy()

        # Annoatation validations rules
        # 1 - score minimal
        # 2 - near group with hierarchical exclusions
        # 3 - exclude duplicate unique family
        # 3 - child with parent

        # Capture validations rules
        # One of annotation respect this rules :
        # 1 - Family tracked and moved
        # 2 - Family trigger or Zone trigger
        # 3 - Zone not ignored

        # tracked : check old position & capture movement
        # trigger : trigger capture
        # abstract : valid if is parent of other annotation
        # unique : only one family annotation by capture
        # zoned : check annotation zone

        annotations = list()

        for result in results:
            annotation = Annotation(
                result,
                self.families_dict, self.zones,
                self.supervisor
            )

            annotations.append(annotation)

        annotations = ArrayHelper.sort(
            annotations, lambda a1, a2: a2.family.index - a1.family.index
        )

        annotations_grouped: list[Annotation] = list()
        for_index_grouped: list[int] = list()
        family_id_grouped: list[int] = list()

        for index, annotation in enumerate(annotations):
            if index not in for_index_grouped and \
                    (not annotation.family.is_unique or annotation.family.id not in family_id_grouped):
                annotation_copy = copy.deepcopy(annotation)
                for _index, _annotation in enumerate(annotations):
                    if _index != index and _index not in for_index_grouped:
                        if annotation_copy.is_near(_annotation):
                            annotation_copy.add_parent(_annotation)
                            for_index_grouped.append(_index)

                annotations_grouped.append(annotation_copy)

                if annotation.family.is_unique:
                    family_id_grouped.append(annotation.family.id)

        for annotation in annotations_grouped:
            if annotation.is_valid():
                self.annotations.append(annotation)

                self.is_triggered = self.is_trigger_annotation(annotation)

                if annotation.parent:
                    self.annotations.append(annotation.parent)

                    self.is_triggered = self.is_trigger_annotation(annotation.parent)

        self.annotations = ArrayHelper.sort(
            self.annotations, lambda a1, a2: a1.family.index - a2.family.index
        )

        for annotation in self.annotations:
            frame_copy = annotation.trace(frame_copy)

        return frame_copy

    def is_trigger_annotation(self, annotation: Annotation):
        if annotation.family.is_tracked:
            first_detection = annotation.family.id not in self.last_detections_dict
            last_detection = self.last_detections_dict.get(annotation.family.id, None)
            coords = (annotation.norm_x_center, annotation.norm_y_center)

            if (
                    first_detection
                    or last_detection is not None and annotation.is_move(last_detection)
            ):
                self.last_detections_dict[annotation.family.id] = coords
                annotation.trigger = Detection.Triggers.MOVE
                return True

        if annotation.zone is not None and cast(Zone, annotation.zone).is_trigger:
            annotation.trigger = Detection.Triggers.ZONE
        elif annotation.family.is_trigger:
            annotation.trigger = Detection.Triggers.FAMILY

        return self.is_triggered or \
            (
                    annotation.trigger == Detection.Triggers.ZONE or
                    annotation.trigger == Detection.Triggers.FAMILY
            ) and \
            (annotation.zone is None or not cast(Zone, annotation.zone).is_ignored)

    def save(self):
        capture = Capture()
        capture.write(self.frame, self.date_capture, self.annotations)
