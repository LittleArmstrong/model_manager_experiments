from __future__ import annotations
import sys
from typing import Any
from ultralytics.utils.loss import v8DetectionLoss, E2ELoss
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn import DetectionModel
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG, RANK
from src.loss.bbox_loss import BboxLoss


class CustomDetectionLoss(v8DetectionLoss):
    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None, loss_func: str | None = None): 
        super().__init__(model, tal_topk,tal_topk2)
        self.bbox_loss = BboxLoss(self.reg_max, loss_func=loss_func).to(self.device)
        # print("CustomDetectionLoss Init!", file=sys.stderr)


class CustomDetectionModel(DetectionModel):
    def __init__(self, cfg="yolo26n.yaml", ch=3, nc=None, verbose=True, loss_func:str|None = None):
        super().__init__(cfg, ch, nc, verbose)
        self.loss_func = loss_func

    def init_criterion(self): 
        """Initialize the loss criterion for the DetectionModel.""" 
        return E2ELoss(self) if getattr(self, "end2end", False) else CustomDetectionLoss(self, loss_func=self.loss_func)


class CustomDetectionTrainer(DetectionTrainer):

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None, loss_func:str|None=None):
        super().__init__(cfg, overrides, _callbacks)
        self.loss_func = loss_func

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        """Return a YOLO detection model.

        Args:
            cfg (str, optional): Path to model configuration file.
            weights (str, optional): Path to model weights.
            verbose (bool): Whether to display model information.

        Returns:
            (DetectionModel): YOLO detection model.
        """
        model = CustomDetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1, loss_func=self.loss_func)
        if weights:
            model.load(weights)
        return model
    

def make_trainer(loss_func):
    """
    Create a CustomDetectionTrainer factory with a fixed IoU loss function.

    :param loss_func: IoU variant to use for bounding box regression.
                      Possible values: "giou", "diou","ciou", "siou", "eiou", "wiou"
    :return: Callable trainer factory compatible with model.train(..., trainer=...)
    """
    def trainer_factory(**kwargs):
        return CustomDetectionTrainer(**kwargs, loss_func=loss_func)
    return trainer_factory
