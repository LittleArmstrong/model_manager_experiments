import torch
import shutil
import random
from pathlib import Path
import cv2
import albumentations as A
from ultralytics import YOLO
import pandas as pd
import re
from copy_paste import copy_paste_objects
from mosaic import create_mosaic
import numpy as np

SEED = 7428321

def get_device():
    return "0" if torch.cuda.is_available() else "cpu"

DEVICE = get_device()

random.seed(SEED)
torch.manual_seed(SEED)

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
    extra_args: dict | None = None,
):
    model = YOLO("YOLOv12n.pt")

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
        device=DEVICE,
        seed=SEED,
        save=True,
        plots=True,
        save_dir=(Path(project)/run_name).resolve(),
        # project=project,
        # name=run_name,
        exist_ok=False,
        verbose=True,
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



def testfall_1_baseline(data, project):
    # project = "runs/testfall_1_baseline"
    # data = "datasets/strawberry_test/data.yaml"
    train_run(
        run_name=f"baseline_run_1",
        data_yaml=data,
        project=project,
    )


def testfall_2_hyperparameter(data, project):
    # project = "runs/testfall_2_hyperparameter"
    # data = "datasets/strawberry_test/data.yaml"

    # Box Loss Weight
    for val in [10, 12, 15]:
        train_run(
            run_name=f"box_loss_{val}",
            data_yaml=data,
            project=project,
            extra_args={"box": val},
        )

    # Distribution Focal Loss
    for val in [2, 3, 4]:
        train_run(
            run_name=f"dfl_{val}",
            data_yaml=data,
            project=project,
            extra_args={"dfl": val},
        )
# def copy_paste_objects(img, boxes, class_labels, labels_dir, images_dir, object_count_max=1):
#     """
#     Füge zufällig Objekte von anderen Originalbildern ein.
#     - img: aktuelles Bild
#     - boxes, class_labels: aktuelle BBoxes
#     - labels_dir: Path zu Labels des Original-Datensatzes
#     - images_dir: Path zu den Originalbildern
#     - object_count_max: max Anzahl Objekte pro Bild
#     """
#     # Alle Originalbilder, die nicht _aug_<zahl> sind
#     original_images = [
#         p for p in images_dir.glob("*.*")
#         if not re.search(r"_aug_\d+\.\w+$", p.name) 
#     ]
#     if not original_images:
#         return img, boxes, class_labels

#     h_target, w_target = img.shape[:2]
#     added_boxes = []
#     added_labels = []

#     # zufällige Anzahl Objekte auswählen
#     num_objects = random.randint(1, object_count_max)

#     # Liste um bereits gewählte Paste-Bilder zu tracken (kein doppelt)
#     used_images = []

#     for _ in range(num_objects):
#         # wähle zufälliges Bild, das noch nicht verwendet wurde
#         candidates = [p for p in original_images if p not in used_images]
#         if not candidates:
#             break
#         paste_img_path = random.choice(candidates)
#         used_images.append(paste_img_path)

#         paste_label_path = labels_dir / f"{paste_img_path.stem}.txt"
#         if not paste_label_path.exists():
#             continue

#         # Paste-Bild laden
#         paste_img = cv2.imread(str(paste_img_path))

#         # Paste-BBoxes und Labels laden
#         paste_boxes = []
#         paste_class_labels = []
#         with open(paste_label_path, "r") as f:
#             for line in f.readlines():
#                 parts = line.strip().split()
#                 cls = int(float(parts[0]))  # cast auf int falls 0.0
#                 bbox = [float(x) for x in parts[1:5]]
#                 paste_boxes.append(bbox)
#                 paste_class_labels.append(cls)

#         if not paste_boxes:
#             continue

#         # zufällige BBox auswählen (wenn nur eine vorhanden, dann diese)
#         idx = random.randint(0, len(paste_boxes) - 1)
#         px, py, pw, ph = paste_boxes[idx]
#         cls = paste_class_labels[idx]

#         # YOLO -> Pixel-Koordinaten
#         h, w = paste_img.shape[:2]
#         x1 = int((px - pw / 2) * w)
#         y1 = int((py - ph / 2) * h)
#         x2 = int((px + pw / 2) * w)
#         y2 = int((py + ph / 2) * h)

#         # Crop aus Paste-Bild
#         obj_crop = paste_img[y1:y2, x1:x2]
#         obj_h, obj_w = obj_crop.shape[:2]

#         # Random Position im Originalbild
#         if obj_h >= h_target or obj_w >= w_target:
#             continue  # Skip wenn Objekt zu groß

#         x_offset = random.randint(0, w_target - obj_w)
#         y_offset = random.randint(0, h_target - obj_h)

#         # Einfügen (overwrite)
#         img[y_offset:y_offset+obj_h, x_offset:x_offset+obj_w] = obj_crop

#         # Neue BBox in YOLO-Normalized-Koordinaten
#         new_px = (x_offset + obj_w / 2) / w_target
#         new_py = (y_offset + obj_h / 2) / h_target
#         new_pw = obj_w / w_target
#         new_ph = obj_h / h_target

#         added_boxes.append([new_px, new_py, new_pw, new_ph])
#         added_labels.append(cls)

#     # Alte BBoxes prüfen und entfernen, falls mehr als 80% von neuen Boxen verdeckt
#     final_boxes = []
#     final_labels = []

#     def iou(box1, box2):
#         # YOLO-normalisierte Box [x, y, w, h] -> Pixel
#         x1a = (box1[0] - box1[2]/2) * w_target
#         y1a = (box1[1] - box1[3]/2) * h_target
#         x2a = (box1[0] + box1[2]/2) * w_target
#         y2a = (box1[1] + box1[3]/2) * h_target

#         x1b = (box2[0] - box2[2]/2) * w_target
#         y1b = (box2[1] - box2[3]/2) * h_target
#         x2b = (box2[0] + box2[2]/2) * w_target
#         y2b = (box2[1] + box2[3]/2) * h_target

#         xi1 = max(x1a, x1b)
#         yi1 = max(y1a, y1b)
#         xi2 = min(x2a, x2b)
#         yi2 = min(y2a, y2b)
#         inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
#         box1_area = (x2a - x1a) * (y2a - y1a)
#         return inter_area / (box1_area + 1e-6)  # IoU in Bezug auf alte Box

#     # check alte Boxen
#     for b, c in zip(boxes, class_labels):
#         if all(iou(b, nb) < 0.8 for nb in added_boxes):
#             final_boxes.append(b)
#             final_labels.append(c)

#     # neue Boxen hinzufügen
#     final_boxes.extend(added_boxes)
#     final_labels.extend(added_labels)

#     return img, final_boxes, final_labels


AUGMENTATIONS = {
    #  "copy_paste": [
    #      {"object_count":1},
    #      {"object_count":2},
    #      {"object_count":3},
    #  ],

    # --- Mosaic (2×2 Grid) ---
    # Albumentations ‚Mosaic‘ braucht zusätzliche Bilder als Metadata:
    "mosaic": [
        {"grid": (2,2)},
        {"grid": (3,3)},
        {"grid": (4,4)},
    ],

    # # --- HSV getrennt: H only ---
    # "hsv-hue": [
    #     A.Compose([A.HueSaturationValue(
    #         hue_shift_limit=int(0.01 * 180),
    #         sat_shift_limit=0,
    #         val_shift_limit=0,
    #         p=1.0
    #     )], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])),
    #     A.Compose([A.HueSaturationValue(
    #         hue_shift_limit=int(0.015 * 180),
    #         sat_shift_limit=0,
    #         val_shift_limit=0,
    #         p=1.0
    #     )], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])),
    #     A.Compose([A.HueSaturationValue(
    #         hue_shift_limit=int(0.02 * 180),
    #         sat_shift_limit=0,
    #         val_shift_limit=0,
    #         p=1.0
    #     )], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])),
    # ],

    # # --- HSV getrennt: S only ---
    # "hsv-saturation": [
    #     A.Compose([A.HueSaturationValue(
    #         hue_shift_limit=0,
    #         sat_shift_limit=int(0.5 * 255),
    #         val_shift_limit=0,
    #         p=1.0
    #     )], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])),
    #     A.Compose([A.HueSaturationValue(
    #         hue_shift_limit=0,
    #         sat_shift_limit=int(0.7 * 255),
    #         val_shift_limit=0,
    #         p=1.0
    #     )], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])),
    #     A.Compose([A.HueSaturationValue(
    #         hue_shift_limit=0,
    #         sat_shift_limit=int(0.9 * 255),
    #         val_shift_limit=0,
    #         p=1.0
    #     )], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])),
    # ],

    # # --- HSV getrennt: V only ---
    # "hsv-value": [
    #     A.Compose([A.HueSaturationValue(
    #         hue_shift_limit=0,
    #         sat_shift_limit=0,
    #         val_shift_limit=int(0.3 * 255),
    #         p=1.0
    #     )], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])),
    #     A.Compose([A.HueSaturationValue(
    #         hue_shift_limit=0,
    #         sat_shift_limit=0,
    #         val_shift_limit=int(0.4 * 255),
    #         p=1.0
    #     )], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])),
    #     A.Compose([A.HueSaturationValue(
    #         hue_shift_limit=0,
    #         sat_shift_limit=0,
    #         val_shift_limit=int(0.5 * 255),
    #         p=1.0
    #     )], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])),
    # ],

    # # --- Crop ---
    # "crop": [
    #     A.Compose([
    #         A.RandomResizedCrop(
    #             height=640, width=640,
    #             scale=(0.85, 0.95),  # Bereich des Bildes, der behalten wird
    #             ratio=(0.9, 1.1),
    #             interpolation=cv2.INTER_LINEAR,
    #             p=1.0
    #         )
    #     ], bbox_params=A.BboxParams(
    #         format="yolo",
    #         label_fields=["class_labels"],
    #         min_visibility=0.3  # entfernt kleine BBoxes
    #     )),

    #     A.Compose([
    #         A.RandomResizedCrop(
    #             height=640, width=640,
    #             scale=(0.75, 0.85),
    #             ratio=(0.9, 1.1),
    #             interpolation=cv2.INTER_LINEAR,
    #             p=1.0
    #         )
    #     ], bbox_params=A.BboxParams(
    #         format="yolo",
    #         label_fields=["class_labels"],
    #         min_visibility=0.3
    #     )),

    #     A.Compose([
    #         A.RandomResizedCrop(
    #             height=640, width=640,
    #             scale=(0.65, 0.75),
    #             ratio=(0.9, 1.1),
    #             interpolation=cv2.INTER_LINEAR,
    #             p=1.0
    #         )
    #     ], bbox_params=A.BboxParams(
    #         format="yolo",
    #         label_fields=["class_labels"],
    #         min_visibility=0.3
    #     )),
    # ],

    # # --- Random Occlusion / CoarseDropout ---
    # "occlusion": [
    #     A.Compose([A.CoarseDropout(max_holes=2, max_height=20, max_width=20, fill_value=0, p=1.0)],
    #               bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])),
    #     A.Compose([A.CoarseDropout(max_holes=2, max_height=40, max_width=40, fill_value=0, p=1.0)],
    #               bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])),
    #     A.Compose([A.CoarseDropout(max_holes=2, max_height=60, max_width=60, fill_value=0, p=1.0)],
    #               bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])),
    # ],
}



def create_augmented_dataset(original_dataset_dir: str, output_dir: str, transform: A.BasicTransform, aug_name:str):
    """
    Erstellt einen neuen augmentierten Datensatz basierend auf einem vorhandenen.
    Nur Bilder in train, die auf '_aug_<zahl>.<ext>' enden, werden augmentiert.
    Labels werden korrekt transformiert. Andere Dateien bleiben unverändert.

    Unterstützt zwei gängige Strukturen:
    1. images/train + labels/train
    2. train/images + train/labels
    """
    original_dataset_dir = Path(original_dataset_dir)
    output_dir = Path(output_dir)

    # -----------------------
    # 1. Datensatz kopieren
    # -----------------------
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(original_dataset_dir, output_dir)

    # -----------------------
    # 2. Train-Bilder augmentieren
    # -----------------------
    # Prüfen welche Struktur vorhanden ist
    if (output_dir / "images" / "train").exists() and (output_dir / "labels" / "train").exists():
        source_images_train_dir = original_dataset_dir / "images" / "train"
        source_labels_train_dir = original_dataset_dir / "labels" / "train"
        target_images_train_dir = output_dir / "images" / "train"
        target_labels_train_dir = output_dir / "labels" / "train"
    elif (output_dir / "train" / "images").exists() and (output_dir / "train" / "labels").exists():
        source_images_train_dir = original_dataset_dir / "train" / "images"
        source_labels_train_dir = original_dataset_dir / "train" / "labels"
        target_images_train_dir = output_dir / "train" / "images"
        target_labels_train_dir = output_dir / "train" / "labels"
    else:
        raise ValueError("Keine gültige Trainings-Ordnerstruktur gefunden!")

    # Regex für Bilder: img_aug_1.jpg, img_aug_2.png, etc.
    pattern = re.compile(r"_aug_\d+\.\w+$")

    for img_path in source_images_train_dir.glob("*.*"):
        if pattern.search(img_path.name):
            base_name = img_path.stem  # z.B. img1_aug_1

            # Bild laden
            img = cv2.imread(str(img_path))

            # Label laden
            label_path = source_labels_train_dir / f"{base_name}.txt"
            if label_path.exists():
                with open(label_path, "r") as f:
                    boxes = []
                    class_labels = []
                    for line in f.readlines():
                        parts = line.strip().split()
                        cls = int(parts[0])
                        bbox = [float(x) for x in parts[1:5]]  # YOLO x_center y_center w h
                        boxes.append(bbox)
                        class_labels.append(cls)
            else:
                boxes = []
                class_labels = []

            # Augmentation anwenden
            if aug_name == "copy_paste":
                object_count_max = transform.get("object_count", 1)  # aus AUGMENTATIONS
                img_new, boxes_new, class_labels_new = copy_paste_objects(
                    img, boxes, class_labels, source_labels_train_dir, source_images_train_dir, object_count_max=object_count_max
                )
                transformed = {"image": img_new, "bboxes": boxes_new, "class_labels": class_labels_new}
            elif aug_name == "mosaic":
                grid = transform.get("grid")  # aus AUGMENTATIONS
                img_new, boxes_new, class_labels_new = create_mosaic(
                    img, boxes, class_labels,source_images_train_dir, source_labels_train_dir, grid=grid
                )
                transformed = {"image": img_new, "bboxes": boxes_new, "class_labels": class_labels_new}
            else:
                transformed = transform(image=img, bboxes=boxes, class_labels=class_labels)

            # Bild speichern 
            target_img_path = target_images_train_dir / img_path.name
            cv2.imwrite(str(target_img_path), transformed["image"])

            # Labels speichern
            target_label_path = target_labels_train_dir / label_path.name
            with open(target_label_path, "w") as f:
                for cls, bbox in zip(transformed["class_labels"], transformed["bboxes"]):
                    bbox_str = " ".join([str(round(x, 6)) for x in bbox])
                    f.write(f"{cls} {bbox_str}\n")

    print(f"[INFO] Augmentierter Datensatz erstellt: {output_dir}")



def testfall_3_training(original_dataset_dir: str, project_dir: str):
    """
    Iteriert über alle Augmentierungen in AUGMENTATIONS,
    erzeugt jeweils einen augmentierten Datensatz und trainiert YOLOv12n darauf.
    """

    original_dataset_dir = Path(original_dataset_dir)
    project_dir = Path(project_dir)

    for aug_name, transforms in AUGMENTATIONS.items():
        for idx, transform in enumerate(transforms, start=1):

            # -------------------------------
            # 1. Neuen Datensatz erstellen
            # -------------------------------
            dataset_name = f"{aug_name}_var{idx}"
            dataset_dir = project_dir / "datasets" / dataset_name
            dataset_dir.mkdir(parents=True, exist_ok=True)

            print(f"[INFO] Erstelle Datensatz: {dataset_name}")
            # TODO: create_augmented_dataset muss implementiert sein
            create_augmented_dataset(
                original_dataset_dir=original_dataset_dir,
                output_dir=dataset_dir,
                transform=transform,
                aug_name=aug_name
            )

            # -------------------------------
            # 2. Trainieren
            # -------------------------------
            run_name = f"{dataset_name}_run"
            data_yaml = dataset_dir / "data.yaml"  # jedes Dataset sollte eigene YAML haben

            print(f"[INFO] Starte Training: {run_name}")
            train_run(
                run_name=run_name,
                data_yaml=str(data_yaml),
                project=str(project_dir),
            )


if __name__ == "__main__":
    # project1 = "runs/testfall_1_baseline"
    # data1 = "datasets/strawberry_test/data.yaml"
    # testfall_1_baseline( data1, project1)

    # project2 = "runs/testfall_2_hyperparameter"
    # data2 = "datasets/strawberry_test/data.yaml"   
    # testfall_2_hyperparameter(data2, project2)

    project3 = "runs/testfall_3_augmentations"
    data3 = "datasets/strawberry_test_augmented"   
    testfall_3_training(data3, project3)