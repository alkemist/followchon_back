from torch.nn import Module


class CombinedModel(Module):
    def __init__(self, models):
        super(CombinedModel, self).__init__()
        self.models = models

    def forward(self, frame):
        preds = []

        for model in self.models:
            preds = preds + model(frame)

        return preds
