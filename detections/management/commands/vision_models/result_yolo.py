import math

from utils.yolo import YoloHelper


class Result_yolo:

    def __init__(self, cls, score, ref_width, ref_height):
        self.cls = cls
        self.score = score
        self.ref_width = ref_width
        self.ref_height = ref_height

        self.norm_x_center = None
        self.norm_y_center = None
        self.norm_width = None
        self.norm_height = None
        self.ortho_tl_x = None
        self.ortho_tl_y = None
        self.ortho_br_x = None
        self.ortho_br_y = None

    def import_hailo_without_padding(self, padded_size: (int, int), padding: (int, int), tl_x, tl_y, br_x, br_y):
        tl_x_ortho = tl_x * padded_size[0]
        tl_x_ortho_trunc = tl_x_ortho - padding[0]
        tl_x = tl_x_ortho_trunc / (padded_size[0] - (padding[0] * 2))

        br_x_ortho = br_x * padded_size[0]
        br_x_ortho_trunc = br_x_ortho - padding[0]
        br_x = br_x_ortho_trunc / (padded_size[0] - (padding[0] * 2))

        tl_y_ortho = tl_y * padded_size[1]
        tl_y_ortho_trunc = tl_y_ortho - padding[1]
        tl_y = tl_y_ortho_trunc / (padded_size[1] - (padding[1] * 2))

        br_y_ortho = br_y * padded_size[1]
        br_y_ortho_trunc = br_y_ortho - padding[1]
        br_y = br_y_ortho_trunc / (padded_size[1] - (padding[1] * 2))

        self.import_hailo(tl_x, tl_y, br_x, br_y)

    def import_hailo(self, tl_x, tl_y, br_x, br_y):
        self.norm_width = max(0, br_x - tl_x)
        self.norm_height = max(0, br_y - tl_y)
        self.norm_x_center = max(0, tl_x + self.norm_width / 2)
        self.norm_y_center = max(0, tl_y + self.norm_height / 2)

        self.ortho_tl_x = math.floor(tl_x * self.ref_width)
        self.ortho_tl_y = math.floor(tl_y * self.ref_height)
        self.ortho_br_x = math.ceil(br_x * self.ref_width)
        self.ortho_br_y = math.ceil(br_y * self.ref_height)

    def import_yolo(self, x_center, y_center, width, height):
        self.norm_x_center = max(0, x_center)
        self.norm_y_center = max(0, y_center)
        self.norm_width = max(0, width)
        self.norm_height = max(0, height)

        ortho = YoloHelper.calc_orthogonal_points(
            x_center,
            y_center,
            width,
            height,
            self.ref_width,
            self.ref_height,
        )

        self.ortho_tl_x = ortho['tl_x']
        self.ortho_tl_y = ortho['tl_y']
        self.ortho_br_x = ortho['br_x']
        self.ortho_br_y = ortho['br_y']

    def import_ortho(self, tl_x, tl_y, br_x, br_y):
        self.ortho_tl_x = tl_x
        self.ortho_tl_y = tl_y
        self.ortho_br_x = br_x
        self.ortho_br_y = br_y

        yolo = YoloHelper.calc_yolo_points(
            tl_x, tl_y,
            br_x, br_y,
            self.ref_width,
            self.ref_height,
        )

        self.norm_x_center = yolo['x_center']
        self.norm_y_center = yolo['y_center']
        self.norm_width = yolo['width']
        self.norm_height = yolo['height']

    def clone(self, cls, score):
        clone = Result_yolo(
            cls,
            score,
            self.ref_width,
            self.ref_height
        )

        clone.norm_width = self.norm_width
        clone.norm_height = self.norm_height
        clone.norm_x_center = self.norm_x_center
        clone.norm_y_center = self.norm_y_center

        clone.ortho_tl_x = self.ortho_tl_x
        clone.ortho_tl_y = self.ortho_tl_y
        clone.ortho_br_x = self.ortho_br_x
        clone.ortho_br_y = self.ortho_br_y

        return clone

    def to_array(self):
        return [
            str(self.cls),
            str(self.score),
            str(self.ref_width),
            str(self.ref_height),
            ' => ',
            str(self.norm_x_center),
            str(self.norm_y_center),
            str(self.norm_width),
            str(self.norm_height),
            ' / ',
            str(self.ortho_tl_x),
            str(self.ortho_tl_y),
            str(self.ortho_br_x),
            str(self.ortho_br_y),
        ]
