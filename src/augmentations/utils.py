from pathlib import Path

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