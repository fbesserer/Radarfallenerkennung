import time
from typing import List, Dict

from pprint import pprint
from torch import Tensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision


# todo speichern in datei umsetzen


def evaluate(preds: List[Dict[str, Tensor]], targets: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    metric: MeanAveragePrecision = MeanAveragePrecision(iou_type="bbox")
    metric.update(preds, targets)
    result: Dict[str, Tensor] = metric.compute()
    pprint(result)
    return result