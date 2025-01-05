import math


class YoloHelper:

    @staticmethod
    def calc_yolo_points(x1, y1, x2, y2, w_img, h_img):
        (w_box, h_box) = x2 - x1, y2 - y1
        return {
            'x_center': max(0, (x1 + (w_box / 2)) / w_img),
            'y_center': max(0, (y1 + (h_box / 2)) / h_img),
            'width': max(0, w_box / w_img),
            'height': max(0, h_box / h_img),
        }

    @staticmethod
    def calc_orthogonal_points(x_center_norm, y_center_norm, w_norm, h_norm, w_img, h_img):
        (w_box, h_box) = w_norm * w_img, h_norm * h_img
        return {
            'tl_x': math.floor((x_center_norm * w_img) - (w_box / 2)),
            'tl_y': math.floor((y_center_norm * h_img) - (h_box / 2)),
            'br_x': math.ceil(x_center_norm * w_img + w_box / 2),
            'br_y': math.ceil(y_center_norm * h_img + h_box / 2),
        }
