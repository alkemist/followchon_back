import math

from ..helpers.yolo import YoloHelper
from ..helpers.image import ImageHelper


class Annotation:
    w_img = 0
    h_img = 0
    labels = list()
    zones = list()

    @staticmethod
    def set_image_size(_w_img, _h_img):
        Annotation.w_img = _w_img
        Annotation.h_img = _h_img

    @staticmethod
    def set_labels(_labels):
        Annotation.labels = _labels

    @staticmethod
    def set_zones(_zones):
        Annotation.zones = _zones

    def __init__(self, box):
        x1, y1, x2, y2 = box.xyxy[0]

        self.conf = math.ceil((box.conf[0] * 100)) / 100
        self.cls = int(box.cls[0])
        self.label = Annotation.labels[self.cls]

        self.coord_point_tl = (float(x1), float(y1))
        self.coord_point_br = (float(x2), float(y2))

        yolo_points = YoloHelper.calc_yolo_points(
            self.coord_point_tl[0], self.coord_point_tl[1],
            self.coord_point_br[0], self.coord_point_br[1],
            Annotation.w_img, Annotation.h_img
        )

        self.norm_center = (
            yolo_points['x_center'],
            yolo_points['y_center']
        )

        self.norm_point_tl = (
            yolo_points['x_center'] - yolo_points['w'] / 2,
            yolo_points['y_center'] - yolo_points['h'] / 2,
        )

        self.norm_point_br = (
            yolo_points['x_center'] + yolo_points['w'] / 2,
            yolo_points['y_center'] + yolo_points['h'] / 2,
        )

        self.line \
            = f"{self.cls} {yolo_points['x_center']} {yolo_points['y_center']} {yolo_points['w']} {yolo_points['h']}\n"

        self.zone = None

        for zone in Annotation.zones:
            if zone.has_point(self.norm_center):
                self.zone = zone
                break

    def trace(self, img):
        return ImageHelper.trace_detected_box_coords(
            img,
            int(self.coord_point_tl[0]),
            int(self.coord_point_tl[1]),
            int(self.coord_point_br[0]),
            int(self.coord_point_br[1]),
            self.label,
            self.conf,
            self.zone.name.value if self.zone else None,
        )
