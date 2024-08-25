from helpers.yolo import YoloHelper


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

    def import_hailo(self, tl_x, tl_y, br_x, br_y):
        self.norm_width = br_x - tl_x
        self.norm_height = br_y - tl_y
        self.norm_x_center = tl_x + self.norm_width / 2
        self.norm_y_center = tl_y + self.norm_height / 2

        self.ortho_tl_x = tl_x * self.ref_width
        self.ortho_tl_y = tl_y * self.ref_height
        self.ortho_br_x = br_x * self.ref_width
        self.ortho_br_y = br_y * self.ref_height

    def import_yolo(self, x_center, y_center, width, height):
        self.norm_x_center = x_center
        self.norm_y_center = y_center
        self.norm_width = width
        self.norm_height = height

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
