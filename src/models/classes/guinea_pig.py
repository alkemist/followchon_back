class GuineaPig:
    cls = 0
    name = 'Guinea Pig'

    def __init__(self):
        self.zone = None

    def set_zone(self, zone):
        if self.zone is None or self.zone.name != zone.name:
            self.zone = zone
            return True
        return False
