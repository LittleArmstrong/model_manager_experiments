from __future__ import annotations
from ultralytics.utils.loss import v8DetectionLoss
from src.loss.bbox_loss import BboxLoss


class CustomDetectionLoss(v8DetectionLoss):
    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None, loss_func: str | None = None): 
        super().__init__(model, tal_topk,tal_topk2)
        self.bbox_loss = BboxLoss(self.reg_max, loss_func=loss_func).to(self.device)
        # print("CustomDetectionLoss Init!", file=sys.stderr)