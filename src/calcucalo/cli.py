from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analyzer import FoodImageAnalyzer
from .dataset import prepare_vietfood67
from .detector import create_detector
from .image_io import load_rgb_image
from .portion import PortionEstimator
from .segmenter import create_segmenter
from .visualize import save_overlay

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRIORS = PROJECT_ROOT / "configs" / "portion_priors.yaml"
DEFAULT_CLASSES = PROJECT_ROOT / "configs" / "vietfood67_classes.yaml"


def _device(value: str) -> str | int:
    return int(value) if value.isdecimal() else value


def run_analyze(args: argparse.Namespace) -> int:
    device = _device(args.device) if args.device is not None else None
    detector = create_detector(
        args.model,
        classes_path=args.classes,
        confidence=args.confidence,
        iou=args.iou,
        image_size=args.image_size,
        device=device,
    )
    segmenter = create_segmenter(args.segmenter, sam_model=args.sam_model, device=device)
    estimator = PortionEstimator(args.priors)
    analyzer = FoodImageAnalyzer(detector, segmenter, estimator)
    result = analyzer.analyze(
        args.image,
        plate_diameter_cm=args.plate_diameter_cm,
        cm_per_pixel=args.cm_per_pixel,
    )
    payload = json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
    if args.output_json:
        target = Path(args.output_json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload, encoding="utf-8")
    else:
        print(payload)
    if args.output_image:
        save_overlay(args.output_image, load_rgb_image(args.image), result)
    return 0


def run_prepare(args: argparse.Namespace) -> int:
    result = prepare_vietfood67(
        args.dataset_root,
        args.output,
        args.classes,
        validate=not args.skip_validation,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    has_errors = any(
        report[metric] > 0
        for report in result["splits"]
        for metric in ("missing_labels", "invalid_lines", "out_of_range_classes", "out_of_range_coordinates")
    )
    return 2 if has_errors else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="calcucalo", description="CalcuCalo vision pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Analyze one food image")
    analyze.add_argument("image", help="Path to JPG/PNG/WebP image")
    analyze.add_argument("--model", required=True, help="YOLO .pt or .onnx weights")
    analyze.add_argument("--segmenter", choices=("sam", "grabcut", "bbox"), default="grabcut")
    analyze.add_argument("--sam-model", default="sam2.1_t.pt")
    analyze.add_argument("--classes", default=str(DEFAULT_CLASSES))
    analyze.add_argument("--priors", default=str(DEFAULT_PRIORS))
    analyze.add_argument("--confidence", type=float, default=0.25)
    analyze.add_argument("--iou", type=float, default=0.60)
    analyze.add_argument("--image-size", type=int, default=640)
    analyze.add_argument("--device", help="cpu, mps or CUDA index such as 0")
    scale = analyze.add_mutually_exclusive_group()
    scale.add_argument("--plate-diameter-cm", type=float)
    scale.add_argument("--cm-per-pixel", type=float)
    analyze.add_argument("--output-json")
    analyze.add_argument("--output-image")
    analyze.set_defaults(handler=run_analyze)

    prepare = subparsers.add_parser("prepare-dataset", help="Validate VietFood67 and write data YAML")
    prepare.add_argument("dataset_root")
    prepare.add_argument("--output", default=str(PROJECT_ROOT / "configs" / "vietfood67.yaml"))
    prepare.add_argument("--classes", default=str(DEFAULT_CLASSES))
    prepare.add_argument("--skip-validation", action="store_true")
    prepare.set_defaults(handler=run_prepare)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
