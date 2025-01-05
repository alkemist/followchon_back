from datetime import datetime
from typing import cast

import cv2
from loguru import logger

from configuration.models import Family, Zone
from detections.management.commands.vision_models.annotation import Annotation
from detections.management.commands.vision_models.source import Source
from detections.management.commands.vision_models.supervisor import Supervisor
from detections.models import Capture, Detection
from utils.array import ArrayHelper


class Capture_analyse:

    def __init__(self, model_version_detect: int, model_version_classify: int,
                 frame: cv2.typing.MatLike, capture_date: datetime, frame_count,
                 last_detections_dict: dict[int, (float, float)],
                 families_dict: dict[int, Family], zones: list[Zone],
                 supervisor: Supervisor
                 ):
        self.model_version_detect = model_version_detect
        self.model_version_classify = model_version_classify
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

        # Capture validations rules
        # One of annotation respect this rules :
        # 1 - Family tracked and moved
        # 2 - Family trigger or Zone trigger
        # 3 - Zone not ignored

        # tracked : check old position & capture movement
        # trigger : trigger capture
        # abstract : valid if is parent of other annotation
        # zoned : check annotation zone

        annotations = list()
        annotations_by_family = dict()

        for result in results:
            annotation = Annotation(
                result,
                self.families_dict, self.zones,
                self.supervisor
            )

            if not annotation.family.is_unique or annotation.family.index not in annotations_by_family:
                if annotation.family.is_unique:
                    annotations_by_family[annotation.family.index] = True

                annotations.append(annotation)
                
                self.is_triggered = self.is_trigger_annotation(annotation)

        if self.supervisor.log_detections_fail:
            for annotation in annotations:
                if annotation.fail is not None:
                    logger.info(
                        f"Class: {annotation.family.index}, "
                        f"Score: {round(annotation.score, 2)}, "
                        f"Fail: {annotation.fail}"
                    )

        self.annotations = ArrayHelper.sort(
            annotations, lambda a1, a2: a1.family.index - a2.family.index
        )

        for annotation in self.annotations:
            frame_copy = annotation.trace(frame_copy)

        return frame_copy

    def is_trigger_annotation(self, annotation: Annotation):
        if self.supervisor.source == Source.PHOTO:
            return True

        if annotation.family.is_tracked or self.supervisor.source == Source.VIDEO:
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
            else:
                annotation.fail = 'stationary'

        if self.supervisor.source == Source.VISION:
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
        else:
            if annotation.fail is None:
                annotation.fail = 'unknown'
            return False

    def save(self):
        Capture().write(self.frame, self.date_capture, self.annotations,
                        self.model_version_detect, self.model_version_classify,
                        self.supervisor.source)
