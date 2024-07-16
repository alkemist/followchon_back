import cv2


class ImageHelper:

    @staticmethod
    def resize_with_ratio(img_origine, width=None, height=None, inter=cv2.INTER_AREA):
        img_copy = img_origine.copy()
        (h, w) = img_copy.shape[:2]

        if width is None and height is None:
            return img_copy

        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))

        return cv2.resize(img_copy, dim, interpolation=inter)

    @staticmethod
    def trace_detected_box_coords(
            img,
            x1,
            y1,
            x2,
            y2,
            label,
            score,
            zone,
            font_scale=2,
            color_text=(0, 255, 0),
            color_bg=(255, 0, 255),
            font_thickness=2,
            font=cv2.FONT_HERSHEY_PLAIN,
    ):
        img_traced = img.copy()
        cv2.rectangle(
            img_traced,
            (x1, y1),
            (x2, y2),
            (255, 0, 255),
            2
        )

        score = str(round(float(score) * 100, 1)) + "%"
        text = "[{}] {}: {}".format(zone, label, score) if zone else "{}: {}".format(label, score)

        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size

        cv2.rectangle(img_traced, (x1, y1), (x1 + text_w + 10, y1 + text_h + 10), color_bg, -1)
        cv2.putText(img_traced, text, (x1 + 5, y1 + text_h + 5 + font_scale - 1), font, font_scale, color_text,
                    font_thickness)

        return img_traced
