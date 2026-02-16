import random
from pathlib import Path
from PIL import Image
import numpy as np
from src.augmentations.utils import get_label_path, load_yolo_labels

def create_mixup(img, boxes, class_labels, images_dir, labels_dir, alpha=0.4):
    """
    Erstellt ein MixUp-Bild aus zwei Bildern und kombiniert die Labels.
    
    Args:
        img: Eingabebild (Pfad, PIL oder numpy array)
        boxes: Bounding Boxes des Eingabebildes (YOLO normalisiert)
        class_labels: Klassenlabels des Eingabebildes
        images_dir: Verzeichnis mit Bildern
        labels_dir: Verzeichnis mit Labels
        alpha: Beta-Verteilungsparameter (typisch 0.2–0.4)
    
    Returns:
        mixup_image, new_boxes, new_class_labels
    """

    # --- Bildtyp prüfen ---
    if isinstance(img, np.ndarray):
        input_img = Image.fromarray(img)
        img_path = None
        return_as_array = True
    elif isinstance(img, Image.Image):
        input_img = img
        img_path = None
        return_as_array = False
    else:
        img_path = Path(img)
        input_img = Image.open(img_path).convert("RGB")
        return_as_array = False

    width, height = input_img.size

    # --- Zufälliges zweites Bild wählen ---
    images_dir_path = Path(images_dir)
    all_images = [p for p in images_dir_path.iterdir() if p.is_file()]

    if img_path and img_path in all_images:
        all_images.remove(img_path)

    if not all_images:
        raise ValueError("Keine weiteren Bilder für MixUp gefunden.")

    second_img_path = random.choice(all_images)
    second_img = Image.open(second_img_path).convert("RGB")

    # Auf gleiche Größe bringen
    second_img = second_img.resize((width, height), Image.Resampling.LANCZOS)

    # --- MixUp Lambda aus Beta-Verteilung ---
    lam = np.random.beta(alpha, alpha)

    # --- Bilder mischen ---
    img1_np = np.array(input_img).astype(np.float32)
    img2_np = np.array(second_img).astype(np.float32)

    mixup_np = lam * img1_np + (1 - lam) * img2_np
    mixup_np = np.clip(mixup_np, 0, 255).astype(np.uint8)

    mixup_image = Image.fromarray(mixup_np)

    # --- Labels laden vom zweiten Bild ---
    label_path = get_label_path(second_img_path, labels_dir)
    second_class_labels, second_boxes = load_yolo_labels(label_path)

    # --- Labels kombinieren ---
    new_class_labels = class_labels + second_class_labels
    new_boxes = boxes + second_boxes

    print(f"MixUp erstellt mit λ={lam:.3f}")
    print(f"Anzahl Objekte: {len(new_class_labels)}")

    if return_as_array:
        return np.array(mixup_image), new_boxes, new_class_labels
    else:
        return mixup_image, new_boxes, new_class_labels
