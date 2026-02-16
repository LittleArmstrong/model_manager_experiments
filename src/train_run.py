from pathlib import Path
from ultralytics import YOLO
import pandas as pd
import torch.nn as nn
from src.custom_yolo import make_trainer

def calc_f1_score(precision:float, recall:float):
    try:
        result = 2 * (precision * recall) / (precision + recall) 
    except:
        result = 0
    return result

def train_run(
    run_name: str,
    data_yaml: str,
    project: str,
    device:str,
    seed:int,
    extra_args: dict | None = None,
    loss_func: str|None = None,
    dropout_rate: float|None = None
):

    model = YOLO("YOLOv12n.pt")
    trainer = make_trainer(loss_func, dropout_rate)

    args = dict(
        data=data_yaml,
        epochs=1,
        imgsz=640,
        batch=16,
        optimizer="SGD",
        lr0=0.01,
        weight_decay=0.0005,
        freeze=9,              
        pretrained=False,
        resume=False,
        device=device,
        seed=seed,
        save=True,
        plots=True,
        save_dir=(Path(project)/run_name).resolve(),
        trainer=trainer,
        exist_ok=False,
        verbose=True
    )

    if extra_args:
        args.update(extra_args)

    yolo_results = model.train(**args)
    class_results = yolo_results.to_df().rows(named=True)
    aggregate = {
        "Class": "Gesamt",
        "Images": "?",
        "Instances": sum(int(r["Instances"]) for r in class_results),
        "Box-P": round(yolo_results.results_dict["metrics/precision(B)"], 5),
        "Box-R": round(yolo_results.results_dict["metrics/recall(B)"], 5),
        "Box-F1": round(
            calc_f1_score(
                yolo_results.results_dict["metrics/precision(B)"],
                yolo_results.results_dict["metrics/recall(B)"]
            ), 5
        ),
        "mAP50": round(yolo_results.results_dict["metrics/mAP50(B)"], 5),
        "mAP50-95": round(yolo_results.results_dict["metrics/mAP50-95(B)"], 5),
    }
    full_results = [aggregate] + class_results
    df = pd.DataFrame(full_results)
    csv_path = Path(project) / run_name / "full_results.csv"
    df.to_csv(
        csv_path,
        index=False,
        sep=";",
        encoding="utf-8"
    )
