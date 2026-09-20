from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class SplitReport:
    split: str
    images: int = 0
    labels: int = 0
    missing_labels: int = 0
    empty_labels: int = 0
    invalid_lines: int = 0
    out_of_range_classes: int = 0
    out_of_range_coordinates: int = 0


def read_class_names(path: str | Path) -> dict[int, str]:
    with Path(path).open("r", encoding="utf-8") as stream:
        values = (yaml.safe_load(stream) or {}).get("names", {})
    if isinstance(values, list):
        names = dict(enumerate(map(str, values)))
    else:
        names = {int(key): str(value) for key, value in values.items()}
    if not names or sorted(names) != list(range(len(names))):
        raise ValueError("Class IDs must be contiguous and start at zero")
    return names


def _find_split(root: Path, split_names: tuple[str, ...]) -> tuple[Path, Path] | None:
    for split in split_names:
        candidates = (
            (root / split / "images", root / split / "labels"),
            (root / "images" / split, root / "labels" / split),
        )
        for image_dir, label_dir in candidates:
            if image_dir.is_dir() and label_dir.is_dir():
                return image_dir, label_dir
    return None


def discover_yolo_layout(root: str | Path) -> dict[str, tuple[Path, Path]]:
    root = Path(root).expanduser().resolve()
    layouts: dict[str, tuple[Path, Path]] = {}
    aliases = {
        "train": ("train", "training"),
        "val": ("val", "valid", "validation"),
        "test": ("test", "testing"),
    }
    for canonical, names in aliases.items():
        found = _find_split(root, names)
        if found:
            layouts[canonical] = found
    if "train" not in layouts or "val" not in layouts:
        raise FileNotFoundError(
            "Could not find YOLO train/val folders. Expected either "
            "<root>/train/images + labels or <root>/images/train + labels/train "
            "(valid/validation are accepted aliases for val)."
        )
    return layouts


def validate_split(
    split: str,
    image_dir: Path,
    label_dir: Path,
    number_of_classes: int,
) -> SplitReport:
    report = SplitReport(split=split)
    images = [path for path in image_dir.rglob("*") if path.suffix.casefold() in IMAGE_SUFFIXES]
    labels = list(label_dir.rglob("*.txt"))
    report.images = len(images)
    report.labels = len(labels)

    label_by_relative_stem = {
        path.relative_to(label_dir).with_suffix("").as_posix().casefold(): path for path in labels
    }
    for image_path in images:
        key = image_path.relative_to(image_dir).with_suffix("").as_posix().casefold()
        label_path = label_by_relative_stem.get(key)
        if label_path is None:
            report.missing_labels += 1
            continue
        lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            report.empty_labels += 1
        for line in lines:
            try:
                values = [float(value) for value in line.split()]
            except ValueError:
                report.invalid_lines += 1
                continue
            if len(values) < 5:
                report.invalid_lines += 1
                continue
            class_id = int(values[0])
            if values[0] != class_id or not 0 <= class_id < number_of_classes:
                report.out_of_range_classes += 1
            # Detection labels have xywh after class; segmentation labels have polygon xy pairs.
            coordinates = values[1:]
            if any(value < 0.0 or value > 1.0 for value in coordinates):
                report.out_of_range_coordinates += 1
            if len(values) != 5 and (len(coordinates) < 6 or len(coordinates) % 2 != 0):
                report.invalid_lines += 1
    return report


def prepare_vietfood67(
    dataset_root: str | Path,
    output_yaml: str | Path,
    classes_yaml: str | Path,
    validate: bool = True,
) -> dict[str, Any]:
    root = Path(dataset_root).expanduser().resolve()
    layouts = discover_yolo_layout(root)
    names = read_class_names(classes_yaml)
    output = Path(output_yaml).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {"path": root.as_posix(), "names": names}
    for split, (image_dir, _) in layouts.items():
        data[split] = image_dir.relative_to(root).as_posix()
    output.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")

    reports: list[dict[str, Any]] = []
    if validate:
        for split, (image_dir, label_dir) in layouts.items():
            reports.append(asdict(validate_split(split, image_dir, label_dir, len(names))))
    return {"dataset_yaml": str(output), "classes": len(names), "splits": reports}

