import random
import cv2
import re


def copy_paste_objects(img, boxes, class_labels, labels_dir, images_dir, object_count_max=1, random_scale=False, scale_range=(0.8, 1.2)):
    """
    Füge zufällig Objekte von anderen Originalbildern ein.

    Args:
        img: np.array, aktuelles Bild
        boxes, class_labels: Listen der aktuellen YOLO-BBoxes und Labels
        labels_dir: Path zu Labels des Original-Datensatzes
        images_dir: Path zu den Originalbildern
        object_count_max: max Anzahl Objekte pro Bild
        random_scale: bool, ob eingefügte Objekte skaliert werden
        scale_range: Tupel (min, max) für Random Scaling
    """
    original_images = [p for p in images_dir.glob("*.*") if not re.search(r"_aug_\d+\.\w+$", p.name)]
    if not original_images:
        return img, boxes, class_labels

    h_target, w_target = img.shape[:2]
    added_boxes = []
    added_labels = []

    # zufällige Anzahl Objekte
    num_objects = random.randint(1, object_count_max)
    used_objects = set()  # (paste_img_path, bbox_idx)
    pasted_pixel_boxes = []  # für Überdeckungs-Check

    for _ in range(num_objects):
        # Objekt wählen, das noch nicht gepastet wurde
        candidates = []
        for p in original_images:
            paste_label_path = labels_dir / f"{p.stem}.txt"
            if paste_label_path.exists():
                with open(paste_label_path, "r") as f:
                    for idx, line in enumerate(f.readlines()):
                        if (p, idx) not in used_objects:
                            candidates.append((p, idx))
        if not candidates:
            break

        paste_img_path, bbox_idx = random.choice(candidates)
        used_objects.add((paste_img_path, bbox_idx))

        # Load paste image + label
        paste_img = cv2.imread(str(paste_img_path))
        with open(labels_dir / f"{paste_img_path.stem}.txt") as f:
            lines = f.readlines()
            parts = lines[bbox_idx].strip().split()
            cls = int(float(parts[0]))
            px, py, pw, ph = [float(x) for x in parts[1:5]]

        h, w = paste_img.shape[:2]
        x1 = int((px - pw / 2) * w)
        y1 = int((py - ph / 2) * h)
        x2 = int((px + pw / 2) * w)
        y2 = int((py + ph / 2) * h)
        obj_crop = paste_img[y1:y2, x1:x2]
        obj_h, obj_w = obj_crop.shape[:2]

        # Random scaling
        if random_scale:
            scale_factor = random.uniform(*scale_range)
            obj_w_new = max(1, int(obj_w * scale_factor))
            obj_h_new = max(1, int(obj_h * scale_factor))
            obj_crop = cv2.resize(obj_crop, (obj_w_new, obj_h_new))
            obj_h, obj_w = obj_crop.shape[:2]

        if obj_h >= h_target or obj_w >= w_target:
            continue

        # max 3 Versuche Position zu finden ohne zu viel Überdeckung
        placed = False
        for _ in range(3):
            x_offset = random.randint(0, w_target - obj_w)
            y_offset = random.randint(0, h_target - obj_h)
            new_pixel_bbox = [x_offset, y_offset, x_offset+obj_w, y_offset+obj_h]

            # Prüfen auf Überdeckung ≥80% mit bereits gepasteten Objekten
            overlap = False
            for prev_bbox in pasted_pixel_boxes:
                xi1 = max(prev_bbox[0], new_pixel_bbox[0])
                yi1 = max(prev_bbox[1], new_pixel_bbox[1])
                xi2 = min(prev_bbox[2], new_pixel_bbox[2])
                yi2 = min(prev_bbox[3], new_pixel_bbox[3])
                inter_area = max(0, xi2-xi1) * max(0, yi2-yi1)
                area_new = (new_pixel_bbox[2]-new_pixel_bbox[0]) * (new_pixel_bbox[3]-new_pixel_bbox[1])
                if inter_area / (area_new + 1e-6) > 0.8:
                    overlap = True
                    break

            if not overlap:
                # einfügen
                img[y_offset:y_offset+obj_h, x_offset:x_offset+obj_w] = obj_crop
                new_px = (x_offset + obj_w / 2) / w_target
                new_py = (y_offset + obj_h / 2) / h_target
                new_pw = obj_w / w_target
                new_ph = obj_h / h_target
                added_boxes.append([new_px, new_py, new_pw, new_ph])
                added_labels.append(cls)
                pasted_pixel_boxes.append(new_pixel_bbox)
                placed = True
                break

        if not placed:
            continue  # konnte Objekt nicht platzieren

    # Alte BBoxes entfernen, falls neue ≥80% überdecken
    def iou(box1, box2):
        x1a = (box1[0] - box1[2]/2) * w_target
        y1a = (box1[1] - box1[3]/2) * h_target
        x2a = (box1[0] + box1[2]/2) * w_target
        y2a = (box1[1] + box1[3]/2) * h_target
        x1b = (box2[0] - box2[2]/2) * w_target
        y1b = (box2[1] - box2[3]/2) * h_target
        x2b = (box2[0] + box2[2]/2) * w_target
        y2b = (box2[1] + box2[3]/2) * h_target
        xi1 = max(x1a, x1b)
        yi1 = max(y1a, y1b)
        xi2 = min(x2a, x2b)
        yi2 = min(y2a, y2b)
        inter_area = max(0, xi2-xi1) * max(0, yi2-yi1)
        box1_area = (x2a-x1a)*(y2a-y1a)
        return inter_area / (box1_area + 1e-6)

    final_boxes = []
    final_labels = []
    for b, c in zip(boxes, class_labels):
        if all(iou(b, nb) < 0.8 for nb in added_boxes):
            final_boxes.append(b)
            final_labels.append(c)

    final_boxes.extend(added_boxes)
    final_labels.extend(added_labels)

    return img, final_boxes, final_labels
