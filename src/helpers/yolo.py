class YoloHelper:

    @staticmethod
    def calc_yolo_points(x1, y1, x2, y2, w_img, h_img):
        (w_box, h_box) = x2 - x1, y2 - y1
        return {
            'x_center': (x1 + (w_box / 2)) / w_img,
            'y_center': (y1 + (h_box / 2)) / h_img,
            'w': w_box / w_img,
            'h': h_box / h_img,
        }

    @staticmethod
    def calc_orthogonal_points(x_center_norm, y_center_norm, w_norm, h_norm, w_img, h_img):
        (w_box, h_box) = w_norm * w_img, h_norm * h_img
        return {
            'x1': round((x_center_norm * w_img) - (w_box / 2)),
            'y1': round((y_center_norm * h_img) - (h_box / 2)),
            'x2': round(x_center_norm * w_img + w_box / 2),
            'y2': round(y_center_norm * h_img + h_box / 2),
        }