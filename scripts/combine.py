import onnx
import torch


class CombinedModel(torch.nn.Module):
    def __init__(self, models):
        super(CombinedModel, self).__init__()
        self.models = models

    def forward(self, frame):
        preds = []

        for model in self.models:
            pred = model.model(frame)

            if isinstance(pred, tuple):
                pred = pred[0]

            preds.append(pred)

        return torch.cat(preds, dim=1)

    def export(self, filename, format="onnx"):
        if format == "onnx":
            dummy_input = torch.randn(1, 3, 1024, 1024)

            torch.onnx.export(
                self,  # Modèle combiné
                dummy_input,  # Entrée fictive
                filename,  # Chemin de sauvegarde
                export_params=True,  # Exporter les paramètres du modèle
                opset_version=11,  # Version d'opset ONNX (ajustez selon les besoins)
                do_constant_folding=True  # Effectuer la simplification des constantes
            )

            onnx_model = onnx.load(filename)

            onnx.checker.check_model(onnx_model)
