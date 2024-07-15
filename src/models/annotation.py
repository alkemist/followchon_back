import math

from ..helpers.image import ImageHelper
from ..helpers.yolo import YoloHelper


class Annotation:
    w_img = 0
    h_img = 0

    @staticmethod
    def set_image_size(_w_img, _h_img):
        Annotation.w_img = _w_img
        Annotation.h_img = _h_img

    def __init__(self, box):
        x1, y1, x2, y2 = box.xyxy[0]

        self.conf = math.ceil((box.conf[0] * 100)) / 100
        self.cls = int(box.cls[0])

        self.coord_point_tl = (float(x1), float(y1))
        self.coord_point_br = (float(x2), float(y2))

        self.yolo_points = YoloHelper.calc_yolo_points(
            self.coord_point_tl[0], self.coord_point_tl[1],
            self.coord_point_br[0], self.coord_point_br[1],
            Annotation.w_img, Annotation.h_img
        )

        self.norm_center = (
            self.yolo_points['x_center'],
            self.yolo_points['y_center']
        )

        self.line \
            = f"{self.cls} {self.yolo_points['x_center']} {self.yolo_points['y_center']} {self.yolo_points['w']} {self.yolo_points['h']}"

    def trace(self, img):
        return ImageHelper.trace_detected_box_coords(
            img,
            int(self.coord_point_tl[0]),
            int(self.coord_point_tl[1]),
            int(self.coord_point_br[0]),
            int(self.coord_point_br[1]),
            '',
            self.conf,
            '',
        )
