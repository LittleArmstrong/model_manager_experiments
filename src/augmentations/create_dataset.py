import shutil
from pathlib import Path
import cv2
import albumentations as A
import re
from src.augmentations.copy_paste import copy_paste_objects
from src.augmentations.mosaic import create_mosaic
from src.augmentations.mixup import create_mixup


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
            elif aug_name == 'mixup':
                alpha = transform.get("alpha")  # aus AUGMENTATIONS
                img_new, boxes_new, class_labels_new = create_mixup(
                    img, boxes, class_labels, source_images_train_dir, source_labels_train_dir, alpha=alpha
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