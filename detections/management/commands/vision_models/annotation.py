import math

import cv2
from typing_extensions import Self

from configuration.models import Family, Zone
from helpers.image import ImageHelper
from helpers.yolo import YoloHelper


class Annotation:

    def __init__(self, box, w_img: int, h_img: int, families_dict: dict[int, Family], zones: list[Zone]):
        x1, y1, x2, y2 = box.xyxy[0]

        self.score = math.ceil((box.conf[0] * 100)) / 100
        family_index = int(box.cls[0])
        self.family = families_dict[family_index]

        self.coord_point_tl = (float(x1), float(y1))
        self.coord_point_br = (float(x2), float(y2))

        self.yolo_points = YoloHelper.calc_yolo_points(
            self.coord_point_tl[0], self.coord_point_tl[1],
            self.coord_point_br[0], self.coord_point_br[1],
            w_img, h_img
        )

        self.zone: Zone | None = None
        self.trigger: str | None = None

        if self.family.is_zoned:
            for zone in zones:
                if zone.has_point((self.yolo_points['x_center'], self.yolo_points['y_center'])):
                    self.zone = zone
                    break

        self.line \
            = (f"{family_index} {self.yolo_points['x_center']} {self.yolo_points['y_center']} " +
               f"{self.yolo_points['w']} {self.yolo_points['h']}")

        self.parent: Annotation | None = None
        self.nears: list[Annotation] = list()
        self.tolerance_margin = (w_img / 10, h_img / 10)

    def is_near(self, annotation: Self):
        return abs(self.coord_point_tl[0] - annotation.coord_point_tl[0]) < self.tolerance_margin[0] and \
            abs(self.coord_point_tl[1] - annotation.coord_point_tl[1]) < self.tolerance_margin[1] and \
            abs(self.coord_point_br[0] - annotation.coord_point_br[0]) < self.tolerance_margin[0] and \
            abs(self.coord_point_br[1] - annotation.coord_point_br[1]) < self.tolerance_margin[1]

    def add_parent(self, annotation: Self):
        if self.family.parent is not None and \
                annotation.family.id == self.family.parent.id and \
                self.parent is None:
            self.parent = annotation
        else:
            self.nears.append(annotation)

    def is_valid(self):
        return not self.family.is_abstract and \
            (self.family.parent is None and self.parent is None or
             self.family.parent is not None and self.parent is not None and
             self.parent.family.id == self.family.parent.id)

    def trace(self, frame: cv2.typing.MatLike):
        return ImageHelper.trace_detected_box_coords(
            frame,
            int(self.coord_point_tl[0]),
            int(self.coord_point_tl[1]),
            int(self.coord_point_br[0]),
            int(self.coord_point_br[1]),
            self.family.name,
            self.score,
            self.zone.name if self.zone is not None else '',
        )

    def to_array(self):
        return [
            self.family.name,
            self.score,
            # (int(self.coord_point_tl[0]), int(self.coord_point_tl[1])),
            # (int(self.coord_point_br[0]), int(self.coord_point_br[1])),
            self.parent.family.name if self.parent else '',
            # [n.family.name for n in self.nears],
            # self.is_valid()
        ]
