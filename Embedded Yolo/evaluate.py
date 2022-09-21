from typing import List, Dict

from pprint import pprint
from torch import Tensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class Metrics:
    def __init__(self, path):
        self.metric: MeanAveragePrecision = MeanAveragePrecision(iou_type="bbox")
        self.path = path

    # todo speichern in datei umsetzen

    def evaluate(self, preds: List[Dict[str, Tensor]], targets: List[Dict[str, Tensor]]) -> None:
        self.metric.update(preds, targets)
        result = self.metric.compute()
        pprint(result)
