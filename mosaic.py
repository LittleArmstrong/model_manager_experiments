import os
import random
from pathlib import Path
from PIL import Image
import numpy as np


def get_label_path(image_path, labels_dir):
    """
    Konvertiert Bildpfad zu Label-Pfad im labels_dir.
    
    Args:
        image_path: Path-Objekt des Bildes
        labels_dir: Verzeichnis wo die Labels liegen
    
    Returns:
        Path-Objekt des Label-Files
    """
    # Nimm den Dateinamen des Bildes und ändere Extension zu .txt
    label_filename = image_path.stem + '.txt'
    label_path = Path(labels_dir) / label_filename
    
    return label_path


def load_yolo_labels(label_path):
    """
    Lädt YOLO-Format Labels aus einer Datei.
    
    Args:
        label_path: Pfad zur Label-Datei
    
    Returns:
        Tuple von (class_labels, boxes) wobei boxes normalisiert sind [x_center, y_center, width, height]
    """
    if not label_path.exists():
        return [], []
    
    class_labels = []
    boxes = []
    
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            class_labels.append(class_id)
            boxes.append([x_center, y_center, width, height])
    
    return class_labels, boxes


def adjust_boxes_for_grid(boxes, row, col, grid_x, grid_y):
    """
    Passt Bounding Boxes für das Grid an.
    
    Args:
        boxes: Liste der Bounding Boxes [x_center, y_center, width, height] (normalisiert 0-1)
        row: Zeile im Grid (0-basiert)
        col: Spalte im Grid (0-basiert)
        grid_x: Anzahl Spalten im Grid
        grid_y: Anzahl Zeilen im Grid
    
    Returns:
        Angepasste Bounding Boxes
    """
    if not boxes:
        return []
    
    adjusted_boxes = []
    
    # Skalierungsfaktoren basierend auf Grid-Größe
    scale_x = 1.0 / grid_x
    scale_y = 1.0 / grid_y
    
    # Offset basierend auf Position im Grid
    offset_x = col * scale_x
    offset_y = row * scale_y
    
    for box in boxes:
        x_center, y_center, width, height = box
        
        # Skaliere die Box
        new_width = width * scale_x
        new_height = height * scale_y
        
        # Passe Center an: skaliere zuerst, dann verschiebe
        new_x_center = (x_center * scale_x) + offset_x
        new_y_center = (y_center * scale_y) + offset_y
        
        adjusted_boxes.append([new_x_center, new_y_center, new_width, new_height])
    
    return adjusted_boxes


def create_mosaic(img, boxes, class_labels, images_dir, labels_dir, img_size=None, grid=(2, 2)):
    """
    Erstellt ein Mosaic-Bild aus mehreren verschiedenen Bildern und passt die Labels an.
    
    Args:
        img: Eingabebild - entweder Pfad (str/Path) oder numpy Array oder PIL Image
        boxes: Bounding Boxes vom Eingabebild (normalisiert) - Liste von [x_center, y_center, width, height]
        class_labels: Klassen-Labels vom Eingabebild - Liste von class IDs
        images_dir: Verzeichnis mit Bildern
        labels_dir: Verzeichnis mit Labels
        img_size: Zielgröße des finalen Bildes (optional, verwendet Originalgröße wenn None)
        grid: Tuple oder Liste (grid_x, grid_y) - Anzahl Bilder auf X und Y Achse (default: (2, 2))
    
    Returns:
        Tuple von (mosaic_image, new_boxes, new_class_labels)
        - mosaic_image: numpy array (gleicher Typ wie Input)
        - new_boxes: Liste der angepassten Bounding Boxes
        - new_class_labels: Liste der Class Labels
    """
    # Grid-Parameter extrahieren
    grid_x, grid_y = grid
    total_images = grid_x * grid_y
    
    # Prüfe ob img ein numpy array, PIL Image oder Pfad ist
    if isinstance(img, np.ndarray):
        # img ist ein numpy array
        input_img = Image.fromarray(img)
        img_path = None
        return_as_array = True
    elif isinstance(img, Image.Image):
        # img ist ein PIL Image
        input_img = img
        img_path = None
        return_as_array = False
    else:
        # img ist ein Pfad
        img_path = Path(img)
        input_img = Image.open(img_path)
        return_as_array = False
    
    # Bestimme img_size wenn nicht angegeben
    if img_size is None:
        img_size = max(input_img.size)  # Nimm die größere Dimension
    
    # Größe pro Bild im Grid
    cell_width = img_size // grid_x
    cell_height = img_size // grid_y
    
    # Finde alle Bilder die auf _aug_<zahl>.<extension> enden
    images_dir_path = Path(images_dir)
    aug_images = []
    
    for img_file in images_dir_path.iterdir():
        if img_file.is_file():
            aug_images.append(img_file) # alle Bilder berücksichtigen
            
            # Prüfe ob Dateiname auf _aug_<zahl>.<extension> endet
            # stem = img_file.stem  # Dateiname ohne Extension
            # if '_aug_' in stem:
            #     parts = stem.split('_aug_')
            #     if len(parts) == 2 and parts[1].isdigit():
            #         aug_images.append(img_file)
    
    # Entferne das Input-Bild aus der Liste falls es ein Pfad ist und darin vorkommt
    if img_path is not None and img_path in aug_images:
        aug_images.remove(img_path)
    
    # Wähle (total_images - 1) zufällige unterschiedliche Bilder aus (da wir das Input-Bild bereits haben)
    num_additional_images = total_images - 1
    if len(aug_images) < num_additional_images:
        raise ValueError(f"Nicht genug aug-Bilder gefunden. Benötigt: {num_additional_images}, Gefunden: {len(aug_images)}")
    
    selected_images = random.sample(aug_images, num_additional_images)
    
    # Listen für die kombinierten Labels
    all_class_labels = []
    all_boxes = []
    
    # Erstelle neues Bild mit Größe img_size x img_size
    mosaic = Image.new('RGB', (img_size, img_size))
    
    # Generiere Positionen für das Grid dynamisch
    positions = []
    for row in range(grid_y):
        for col in range(grid_x):
            x_pos = col * cell_width
            y_pos = row * cell_height
            positions.append((x_pos, y_pos))
    
    # Verarbeite das erste Bild (Input-Bild)
    img_resized = input_img.resize((cell_width, cell_height), Image.Resampling.LANCZOS)
    mosaic.paste(img_resized, positions[0])
    
    # Nutze übergebene boxes und class_labels für das Input-Bild
    row = 0
    col = 0
    adjusted_boxes = adjust_boxes_for_grid(boxes, row, col, grid_x, grid_y)
    all_class_labels.extend(class_labels)
    all_boxes.extend(adjusted_boxes)
    
    # Lade und platziere die restlichen Bilder
    for idx, (img_path_current, pos) in enumerate(zip(selected_images, positions[1:]), start=1):
        # Lade Bild
        img_obj = Image.open(img_path_current)
        
        # Resize auf Grid-Zellengröße
        img_resized = img_obj.resize((cell_width, cell_height), Image.Resampling.LANCZOS)
        
        # Platziere im Mosaic
        mosaic.paste(img_resized, pos)
        
        # Lade Labels von den anderen Bildern
        label_path = get_label_path(img_path_current, labels_dir)
        current_class_labels, current_boxes = load_yolo_labels(label_path)
        
        # Berechne Position im Grid (row, col)
        row = idx // grid_x
        col = idx % grid_x
        
        # Passe Boxes für die Position im Grid an
        adjusted_boxes = adjust_boxes_for_grid(current_boxes, row, col, grid_x, grid_y)
        
        # Füge zu den Gesamt-Labels hinzu
        all_class_labels.extend(current_class_labels)
        all_boxes.extend(adjusted_boxes)
    
    print(f"Mosaic erstellt aus {total_images} Bildern")
    print(f"Anzahl Objekte: {len(all_class_labels)}")
    
    # Konvertiere zurück zu numpy array wenn Input ein numpy array war
    if return_as_array:
        mosaic_array = np.array(mosaic)
        return mosaic_array, all_boxes, all_class_labels
    else:
        return mosaic, all_boxes, all_class_labels


# Beispielaufruf
if __name__ == "__main__":
    # Beispiel-Parameter
    img = "/pfad/zum/bild.jpg"
    
    # Bounding Boxes im YOLO-Format (normalisiert 0-1): [x_center, y_center, width, height]
    boxes = [
        [0.5, 0.5, 0.3, 0.4],  # Beispiel-Box
        [0.2, 0.3, 0.15, 0.2]  # Beispiel-Box
    ]
    
    # Class Labels (entsprechend zu den Boxes)
    class_labels = [0, 1]  # Beispiel: Klasse 0 und Klasse 1
    
    images_dir = "/pfad/zum/images_verzeichnis"
    labels_dir = "/pfad/zum/labels_verzeichnis"
    
    # Erstelle 2x2 Mosaic mit Originalgröße (Standard)
    mosaic_image, new_boxes, new_class_labels = create_mosaic(
        img, boxes, class_labels, images_dir, labels_dir
    )
    
    # Oder mit 3x3 Grid
    # mosaic_image, new_boxes, new_class_labels = create_mosaic(
    #     img, boxes, class_labels, images_dir, labels_dir, grid=(3, 3)
    # )
    
    # Oder mit 2x3 Grid und spezifischer Größe
    # mosaic_image, new_boxes, new_class_labels = create_mosaic(
    #     img, boxes, class_labels, images_dir, labels_dir, img_size=640, grid=(2, 3)
    # )
    
    print(f"\nNeue Labels: {len(new_class_labels)} Objekte")
    for i, (cls, box) in enumerate(zip(new_class_labels, new_boxes)):
        print(f"  Objekt {i+1}: Klasse {cls}, Box {box}")
    
    # Optional: Mosaic-Bild speichern
    # mosaic_image.save("/pfad/zum/output.jpg")