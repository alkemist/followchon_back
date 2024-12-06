from torch.nn import Module


class CombinedYOLOModel(Module):
    def __init__(self, models):
        super(CombinedYOLOModel, self).__init__()
        self.models = models

    def forward(self, frame):
        preds = []

        for model in self.models:
            preds = preds + model(frame)

        return preds
