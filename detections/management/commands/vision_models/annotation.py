import math

from helpers.image import ImageHelper
from helpers.yolo import YoloHelper


class Annotation:

    def __init__(self, box, w_img, h_img, families_dict, zones):
        x1, y1, x2, y2 = box.xyxy[0]

        self.score = math.ceil((box.conf[0] * 100)) / 100
        self.family_index = int(box.cls[0])
        self.family_name = families_dict[self.family_index].name

        self.coord_point_tl = (float(x1), float(y1))
        self.coord_point_br = (float(x2), float(y2))

        self.yolo_points = YoloHelper.calc_yolo_points(
            self.coord_point_tl[0], self.coord_point_tl[1],
            self.coord_point_br[0], self.coord_point_br[1],
            w_img, h_img
        )

        self.zone = None
        for zone in zones:
            if zone.has_point((self.yolo_points['x_center'], self.yolo_points['y_center'])):
                self.zone = zone
                break

        self.line \
            = (f"{self.family_index} {self.yolo_points['x_center']} {self.yolo_points['y_center']} " +
               f"{self.yolo_points['w']} {self.yolo_points['h']}")

    def trace(self, img):
        return ImageHelper.trace_detected_box_coords(
            img,
            int(self.coord_point_tl[0]),
            int(self.coord_point_tl[1]),
            int(self.coord_point_br[0]),
            int(self.coord_point_br[1]),
            self.family_name,
            self.score,
            self.zone.name if self.zone is not None else '',
        )
