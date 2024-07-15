class Zone:

    def __init__(self, name, x_center_norm, y_center_norm, width_norm, height_norm):
        self.name = name
        self.point_tl = (
                x_center_norm - width_norm / 2,
                y_center_norm - height_norm / 2
        )
        self.point_br = (
                x_center_norm + width_norm / 2,
                y_center_norm + height_norm / 2
        )

    def has_point(self, point):
        has_in_x = self.point_tl[0] <= point[0] <= self.point_br[0]
        has_in_y = self.point_tl[1] <= point[1] <= self.point_br[1]
        return has_in_x and has_in_y
