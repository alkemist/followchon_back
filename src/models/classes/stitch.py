from src.models.classes.guinea_pig import GuineaPig


class Stitch(GuineaPig):
    cls = 2
    name = 'Stitch'

    def __init__(self):
        super().__init__()
