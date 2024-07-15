from src.models.classes.guinea_pig import GuineaPig


class Noisette(GuineaPig):
    cls = 1
    name = 'Noisette'

    def __init__(self):
        super().__init__()
