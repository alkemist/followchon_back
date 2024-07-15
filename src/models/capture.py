import os
from datetime import datetime
from pathlib import Path

import cv2

from .annotation import Annotation
from .classes.guinea_pig import GuineaPig
from .classes.noisette import Noisette
from .classes.stitch import Stitch
from ..data.zones import Zones
from ..helpers.array import ArrayHelper
from ..helpers.image import ImageHelper


class Capture:
    cls = list()
    capture_width = 0
    capture_height = 0
    capture_min_conf = 0.8
    norm_diff_duplicate_error = 0.1

    @staticmethod
    def set_classes(cls):
        Capture.cls = cls

    @staticmethod
    def set_capture_size(capture_width, capture_height):
        Capture.capture_width = capture_width
        Capture.capture_height = capture_height

    @staticmethod
    def set_capture_min_conf(capture_min_conf):
        Capture.capture_min_conf = capture_min_conf

    def __init__(self, frame):
        self.annotations = list()
        self.frame = frame

        (h_img, w_img) = frame.shape[:2]
        Annotation.set_image_size(w_img, h_img)

        self.valid = False
        self.cls_counts = {}
        self.cls_zones = {}
        self.log = list()

        for cls in Capture.cls:
            self.cls_counts[cls] = 0
            self.cls_zones[cls] = None

    def detect(self, results):
        frame_copy = self.frame.copy()
        annotations_by_cls = {}
        duplicate = False

        self.annotations = list()
        self.log = list()

        for r in results:
            boxes = r.boxes

            for box in boxes:
                annotation = Annotation(box)

                # Condition 1 : Good detection
                if annotation.conf > Capture.capture_min_conf:

                    # Condition 2 : No duplicate
                    if annotation.cls in annotations_by_cls and len(annotations_by_cls[annotation.cls]) > 0:
                        duplicate = all(
                            annotation.norm_center[0] - center[0] < Capture.norm_diff_duplicate_error
                            for i, center in enumerate(annotations_by_cls[annotation.cls])
                        )

                    if not duplicate:
                        if annotation.cls not in annotations_by_cls:
                            annotations_by_cls[annotation.cls] = list()
                        annotations_by_cls[annotation.cls].append(annotation.norm_center)

                        self.cls_counts[annotation.cls] \
                            = self.cls_counts[annotation.cls] + 1 if annotation.cls in self.cls_counts else 1
                        self.cls_zones[annotation.cls] = annotation.zone

                        self.annotations.append(annotation)
                    else:
                        break

        self.annotations = ArrayHelper.sort(self.annotations, lambda a1, a2: a1.cls - a2.cls)

        for annotation in self.annotations:
            if annotation.zone is not None:
                self.log.append((annotation.label, annotation.zone.name.value, annotation.conf))
            else:
                self.log.append((annotation.label, annotation.conf))
            frame_copy = annotation.trace(frame_copy)

        # Condition : Consistent accounting
        # Condition : Outside areas
        self.valid = \
            (
                    self.cls_counts[Stitch.cls] == 1
                    and self.cls_counts[Noisette.cls] == 1
                    or any([
                annotation.zone is not None and
                (
                        annotation.zone.name == Zones.FONTAINE
                )
                for annotation in self.annotations
            ])
            ) \
            and self.cls_counts[Stitch.cls] <= 1 and self.cls_counts[Noisette.cls] <= 1 \
            and self.cls_counts[GuineaPig.cls] == self.cls_counts[Noisette.cls] + self.cls_counts[Stitch.cls] \
            # and any([
        #     annotation.zone is None or (
        #             annotation.zone.name != Zones.TUNNEL
        #             and annotation.zone.name != Zones.FOIN
        #     )
        #     for annotation in self.annotations
        # ])

        return frame_copy

    def save(self, captures_directory):
        filename = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')

        captures_labels_dir = f"{captures_directory}/labels"
        captures_images_dir = f"{captures_directory}/images"

        if not os.path.exists(captures_images_dir):
            os.makedirs(captures_images_dir)

        cv2.imwrite(
            f'{captures_images_dir}/{filename}.jpg',
            ImageHelper.resize_with_ratio(self.frame, Capture.capture_width, Capture.capture_height)
        )

        annotations = [annotation.line for annotation in self.annotations]
        annotations.sort()

        file = Path(f'{captures_labels_dir}/{filename}.txt')
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("".join(annotations))
