from pathlib import Path
import cv2
from ultralytics.utils.plotting import Annotator

def create_annotated_folder_yolo(dataset_dir: str, output_dir_name="annotated", class_names=None):
    """
    Erstellt annotierte Bilder für einen YOLO-Datensatz, ähnlich wie draw_predictions_on_image.
    Nutzt ultralytics Annotator, liest Labels aus .txt-Dateien.

    Args:
        dataset_dir (str): Pfad zum YOLO-Datensatz (images/train + labels/train)
        output_dir_name (str): Name des Ordners für annotierte Bilder
        class_names (dict): Mapping class_id -> class_name, z.B. {0: "strawberry"}
    """
    dataset_dir = Path(dataset_dir)

    if (dataset_dir / "images" / "train").exists() and (dataset_dir / "labels" / "train").exists():
        images_train_dir = dataset_dir / "images" / "train"
        labels_train_dir = dataset_dir / "labels" / "train"
    elif (dataset_dir / "train" / "images").exists() and (dataset_dir / "train" / "labels").exists():
        images_train_dir = dataset_dir / "train" / "images"
        labels_train_dir = dataset_dir / "train" / "labels"
    else:
        raise ValueError("Keine gültige Trainings-Ordnerstruktur gefunden!")

    if not images_train_dir.exists() or not labels_train_dir.exists():
        raise ValueError("images/train oder labels/train existieren nicht!")

    annotated_dir = dataset_dir / output_dir_name
    annotated_dir.mkdir(parents=True, exist_ok=True)

    if class_names is None:
        class_names = {}

    for img_path in images_train_dir.glob("*.*"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        annotator = Annotator(img)

        label_path = labels_train_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            h, w = img.shape[:2]
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    cls = int(float(parts[0]))
                    x_center, y_center, bw, bh = map(float, parts[1:5])
                    # YOLO -> XYXY
                    x1 = int((x_center - bw / 2) * w)
                    y1 = int((y_center - bh / 2) * h)
                    x2 = int((x_center + bw / 2) * w)
                    y2 = int((y_center + bh / 2) * h)
                    xyxy = [x1, y1, x2, y2]
                    class_label = class_names.get(cls, str(cls))
                    annotator.box_label(xyxy, class_label)

        img_annotated = annotator.result()
        cv2.imwrite(str(annotated_dir / img_path.name), img_annotated)

    print(f"[INFO] Annotierte Bilder gespeichert in: {annotated_dir}")


if __name__ == "__main__":
    # create_annotated_folder_yolo("runs/testfall_3_augmentations/datasets/copy_paste_var3")
    # create_annotated_folder_yolo("runs/testfall_3_augmentations/datasets/mosaic_var1")
    create_annotated_folder_yolo("datasets\Erdbeeren100pC_640px")

