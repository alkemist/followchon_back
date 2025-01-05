import cv2

from configuration.models import Family, Zone
from detections.management.commands.vision_models.result_yolo import Result_yolo
from detections.management.commands.vision_models.supervisor import Supervisor
from utils.image import ImageHelper


class Annotation:

    def __init__(self,
                 result: Result_yolo,
                 families_dict: dict[int, Family],
                 zones: list[Zone],
                 supervisor: Supervisor
                 ):

        self.score = result.score
        self.family = families_dict[result.cls]

        self.ortho_point_tl = (result.ortho_tl_x, result.ortho_tl_y)
        self.ortho_point_br = (result.ortho_br_x, result.ortho_br_y)

        self.norm_x_center = result.norm_x_center
        self.norm_y_center = result.norm_y_center
        self.norm_width = result.norm_width
        self.norm_height = result.norm_height

        self.zone: Zone | None = None
        self.trigger: str | None = None
        self.fail: str | None = None

        if self.family.is_zoned:
            for zone in zones:
                if zone.has_point((result.norm_x_center, result.norm_y_center)):
                    self.zone = zone
                    break

        self.line \
            = (f"{result.cls} {result.norm_x_center} {result.norm_y_center} " +
               f"{result.norm_width} {result.norm_height}")

        self.move_tolerance_margin_norm = supervisor.detection_move_margin_norm

    def is_move(self, coords: (float, float)):
        return abs(self.norm_x_center - coords[0]) > self.move_tolerance_margin_norm \
            or abs(self.norm_y_center - coords[1]) > self.move_tolerance_margin_norm

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
            # [n.family.name for n in self.nears],
        ]
