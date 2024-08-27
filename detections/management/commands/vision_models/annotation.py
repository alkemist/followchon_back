import cv2
from typing_extensions import Self

from configuration.models import Family, Zone
from detections.management.commands.vision_models.result_yolo import Result_yolo
from helpers.image import ImageHelper


class Annotation:

    def __init__(self, result: Result_yolo, families_dict: dict[int, Family], zones: list[Zone]):

        self.score = result.score
        self.family = families_dict[result.cls]

        self.ortho_point_tl = (result.ortho_tl_x, result.ortho_tl_y)
        self.ortho_point_br = (result.ortho_br_x, result.ortho_br_y)

        self.yolo_points = {
            'x_center': result.norm_x_center,
            'y_center': result.norm_y_center,
            'width': result.norm_width,
            'height': result.norm_height,
        }

        self.zone: Zone | None = None
        self.trigger: str | None = None

        # ortho_x_center = result.ortho_tl_x + (result.ortho_br_x - result.ortho_tl_x / 2)
        # ortho_y_center = result.ortho_tl_y + (result.ortho_br_y - result.ortho_tl_y / 2)

        if self.family.is_zoned:
            for zone in zones:
                if zone.has_point((result.norm_x_center, result.norm_y_center)):
                    self.zone = zone
                    break

        self.line \
            = (f"{result.cls} {result.norm_x_center} {result.norm_y_center} " +
               f"{result.norm_width} {result.norm_height}")

        self.parent: Annotation | None = None
        self.nears: list[Annotation] = list()
        self.tolerance_margin = (result.ref_width / 20, result.ref_height / 20)

    def is_near(self, annotation: Self):
        return abs(self.ortho_point_tl[0] - annotation.ortho_point_tl[0]) < self.tolerance_margin[0] and \
            abs(self.ortho_point_tl[1] - annotation.ortho_point_tl[1]) < self.tolerance_margin[1] and \
            abs(self.ortho_point_br[0] - annotation.ortho_point_br[0]) < self.tolerance_margin[0] and \
            abs(self.ortho_point_br[1] - annotation.ortho_point_br[1]) < self.tolerance_margin[1]

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
            int(self.ortho_point_tl[0]),
            int(self.ortho_point_tl[1]),
            int(self.ortho_point_br[0]),
            int(self.ortho_point_br[1]),
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
