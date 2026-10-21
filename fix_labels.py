
from pathlib import Path
import shutil

# Main dataset folder
PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "weed detection -bounding boxes-.v7i.yolov8"

# Backup will be saved outside the dataset's train/valid/test folders
BACKUP_DIR = PROJECT_DIR / "labels_backup_before_bbox_fix"

SPLITS = ["train", "valid", "test"]


def convert_polygon_to_box(values):
    """Convert YOLO polygon coordinates into a YOLO bounding box."""
    coordinates = values[1:]

    if len(coordinates) < 6 or len(coordinates) % 2 != 0:
        raise ValueError("Invalid polygon coordinates")

    x_coords = coordinates[0::2]
    y_coords = coordinates[1::2]

    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    return [
        values[0],
        x_center,
        y_center,
        width,
        height,
    ]


def main():
    label_files = []

    # Check all label files before changing anything
    for split in SPLITS:
        label_folder = DATASET_DIR / split / "labels"

        if not label_folder.exists():
            raise FileNotFoundError(f"Missing label folder: {label_folder}")

        label_files.extend(label_folder.glob("*.txt"))

    converted_count = 0

    # Validate annotation formats first
    for file_path in label_files:
        for line_number, line in enumerate(
            file_path.read_text(encoding="utf-8-sig").splitlines(), start=1
        ):
            if not line.strip():
                continue

            parts = line.split()

            try:
                if len(parts) < 5:
                    raise ValueError("Fewer than 5 values")

                class_id = int(parts[0])
                coordinates = [float(value) for value in parts[1:]]

                if class_id not in (0, 1):
                    raise ValueError(f"Unexpected class ID: {class_id}")

                if len(parts) == 5:
                    if not all(0 <= value <= 1 for value in coordinates):
                        raise ValueError("Bounding box coordinates outside 0–1")
                else:
                    if len(coordinates) < 6 or len(coordinates) % 2 != 0:
                        raise ValueError("Invalid polygon coordinate count")

                    if not all(0 <= value <= 1 for value in coordinates):
                        raise ValueError("Polygon coordinates outside 0–1")

                    converted_count += 1

            except ValueError as error:
                raise ValueError(
                    f"Problem in {file_path}, line {line_number}: {error}"
                ) from error

    print(f"Checked {len(label_files)} label files.")
    print(f"Polygon annotations to convert: {converted_count}")

    # Back up every label file before modifying any of them
    for split in SPLITS:
        source_folder = DATASET_DIR / split / "labels"
        backup_folder = BACKUP_DIR / split / "labels"
        backup_folder.mkdir(parents=True, exist_ok=True)

        for file_path in source_folder.glob("*.txt"):
            shutil.copy2(file_path, backup_folder / file_path.name)

    print(f"Backup created at: {BACKUP_DIR}")

    # Convert polygon rows while keeping existing box rows unchanged
    for file_path in label_files:
        new_lines = []

        for line in file_path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue

            parts = line.split()
            values = [int(parts[0])] + [float(value) for value in parts[1:]]

            if len(parts) > 5:
                values = convert_polygon_to_box(values)

            new_lines.append(
                str(values[0]) + " " +
                " ".join(f"{value:.8f}" for value in values[1:])
            )

        file_path.write_text(
            "\n".join(new_lines) + ("\n" if new_lines else ""),
            encoding="utf-8",
        )

    print("Finished converting polygon annotations to bounding boxes.")
    print("Original label files are backed up. Do not delete the backup yet.")


if __name__ == "__main__":
    main()
