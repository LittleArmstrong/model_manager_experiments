import torch
import random
from pathlib import Path
from src.augmentations.create_dataset import create_augmented_dataset
from src.train_run import train_run
import albumentations as A
import cv2

SEED = 7428321

def get_device():
    return "0" if torch.cuda.is_available() else "cpu"

DEVICE = get_device()

random.seed(SEED)
torch.manual_seed(SEED)


def testfall_1_baseline(data, project):
    train_run(
        run_name=f"baseline_run_1",
        data_yaml=data,
        project=project,
        device=DEVICE,
        seed=SEED
    )


def testfall_2_hyperparameter(data, project):
    # Box Loss Weight
    loss_funcs = ['ciou', 'siou', 'eiou']
    for loss_func in loss_funcs:
        if loss_func != 'ciou':
            train_run(
                run_name=f"{loss_func}",
                data_yaml=data,
                project=project,
                device=DEVICE,
                seed=SEED,
                loss_func=loss_func   
            )
        for val in [10, 12, 15]:
            train_run(
                run_name=f"{loss_func}_box_loss_{val}",
                data_yaml=data,
                project=project,
                extra_args={"box": val},
                device=DEVICE,
                seed=SEED,
                loss_func=loss_func   
            )

        # Distribution Focal Loss
        for val in [2, 3, 4]:
            train_run(
                run_name=f"{loss_func}_dfl_{val}",
                data_yaml=data,
                project=project,
                extra_args={"dfl": val},
                device=DEVICE,
                seed=SEED,
                loss_func=loss_func  
            )

        # class weight loss
        for val in [1, 2, 3]:
            train_run(
                run_name=f"{loss_func}_cls_{val}",
                data_yaml=data,
                project=project,
                extra_args={"cls": val},
                device=DEVICE,
                seed=SEED,
                loss_func=loss_func  
            )

AUGMENTATIONS = {
    # --- CopyPaste  ---
     "copy_paste": [
         {"object_count":1},
         {"object_count":2},
         {"object_count":3},
     ],

    # --- Mosaic ---
    "mosaic": [
        {"grid": (2,2)},
        {"grid": (3,3)},
        {"grid": (4,4)},
    ],

    # --- Mixup ---
    "mixup": [
        {"alpha": 0.2},
        {"alpha": 0.3},
        {"alpha": 0.4},
    ],

    # --- HSV getrennt: H only ---
    "hsv-hue": [
        A.Compose([A.HueSaturationValue(
            hue_shift_limit=int(0.01 * 180),
            sat_shift_limit=0,
            val_shift_limit=0,
            p=1.0
        )], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])),
        A.Compose([A.HueSaturationValue(
            hue_shift_limit=int(0.015 * 180),
            sat_shift_limit=0,
            val_shift_limit=0,
            p=1.0
        )], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])),
        A.Compose([A.HueSaturationValue(
            hue_shift_limit=int(0.02 * 180),
            sat_shift_limit=0,
            val_shift_limit=0,
            p=1.0
        )], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])),
    ],

    # --- HSV getrennt: S only ---
    "hsv-saturation": [
        A.Compose([A.HueSaturationValue(
            hue_shift_limit=0,
            sat_shift_limit=int(0.5 * 255),
            val_shift_limit=0,
            p=1.0
        )], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])),
        A.Compose([A.HueSaturationValue(
            hue_shift_limit=0,
            sat_shift_limit=int(0.7 * 255),
            val_shift_limit=0,
            p=1.0
        )], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])),
        A.Compose([A.HueSaturationValue(
            hue_shift_limit=0,
            sat_shift_limit=int(0.9 * 255),
            val_shift_limit=0,
            p=1.0
        )], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])),
    ],

    # --- HSV getrennt: V only ---
    "hsv-value": [
        A.Compose([A.HueSaturationValue(
            hue_shift_limit=0,
            sat_shift_limit=0,
            val_shift_limit=int(0.3 * 255),
            p=1.0
        )], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])),
        A.Compose([A.HueSaturationValue(
            hue_shift_limit=0,
            sat_shift_limit=0,
            val_shift_limit=int(0.4 * 255),
            p=1.0
        )], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])),
        A.Compose([A.HueSaturationValue(
            hue_shift_limit=0,
            sat_shift_limit=0,
            val_shift_limit=int(0.5 * 255),
            p=1.0
        )], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])),
    ],

    # --- Crop ---
    "crop": [
        A.Compose([
            A.RandomResizedCrop(
                height=640, width=640,
                scale=(0.85, 0.95),  # Bereich des Bildes, der behalten wird
                ratio=(0.9, 1.1),
                interpolation=cv2.INTER_LINEAR,
                p=1.0
            )
        ], bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3  # entfernt kleine BBoxes
        )),

        A.Compose([
            A.RandomResizedCrop(
                height=640, width=640,
                scale=(0.75, 0.85),
                ratio=(0.9, 1.1),
                interpolation=cv2.INTER_LINEAR,
                p=1.0
            )
        ], bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3
        )),

        A.Compose([
            A.RandomResizedCrop(
                height=640, width=640,
                scale=(0.65, 0.75),
                ratio=(0.9, 1.1),
                interpolation=cv2.INTER_LINEAR,
                p=1.0
            )
        ], bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3
        )),
    ],

    # --- Random Occlusion / CoarseDropout ---
    "occlusion": [
        A.Compose([A.CoarseDropout(max_holes=2, max_height=20, max_width=20, fill_value=0, p=1.0)],
                  bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])),
        A.Compose([A.CoarseDropout(max_holes=2, max_height=40, max_width=40, fill_value=0, p=1.0)],
                  bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])),
        A.Compose([A.CoarseDropout(max_holes=2, max_height=60, max_width=60, fill_value=0, p=1.0)],
                  bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])),
    ],
}






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
                device=DEVICE,
                seed=SEED
            )


if __name__ == "__main__":
    # project1 = "runs/testfall_1_baseline"
    # data1 = "datasets/strawberry_test/data.yaml"
    # testfall_1_baseline( data1, project1)

    project2 = "runs/testfall_2_hyperparameter"
    data2 = "datasets/strawberry_test/data.yaml"   
    testfall_2_hyperparameter(data2, project2)

    # project3 = "runs/testfall_3_augmentations"
    # data3 = "datasets/strawberry_test_augmented"   
    # testfall_3_training(data3, project3)