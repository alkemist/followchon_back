class Math:

    @staticmethod
    def is_point_in_rectangle(px, py, rect):
        # Extraire les coordonnées des sommets du rectangle
        x1, y1 = rect[0]
        x2, y2 = rect[1]
        x3, y3 = rect[2]
        x4, y4 = rect[3]

        # Trouver les min et max pour x et y pour définir les limites du rectangle
        min_x = min(x1, x2, x3, x4)
        max_x = max(x1, x2, x3, x4)
        min_y = min(y1, y2, y3, y4)
        max_y = max(y1, y2, y3, y4)

        # Vérifier si le point (px, py) est à l'intérieur des limites
        if min_x <= px <= max_x and min_y <= py <= max_y:
            return True
        return False

    @staticmethod
    def are_points_inside_rectangle(points, rect):
        for (px, py) in points:
            if not Math.is_point_in_rectangle(px, py, rect):
                return False
        return True

    @staticmethod
    def determine_id_range(id_central, id_min, id_max, x):
        # Calculer le nombre d'éléments de chaque côté de l'id_central
        half_range = (x - 1) // 2

        # Initialiser les limites de la plage autour de id_central
        start_id = id_central - half_range
        end_id = id_central + half_range

        # Ajuster les limites si elles sortent des bornes définies par id_min et id_max
        if start_id < id_min:
            start_id = id_min
            end_id = min(id_min + x - 1, id_max)  # Assurer un maximum de x éléments
        if end_id > id_max:
            end_id = id_max
            start_id = max(id_max - x + 1, id_min)  # Assurer un maximum de x éléments

        return start_id, end_id