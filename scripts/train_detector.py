from __future__ import annotations

import argparse
from pathlib import Path


def parse_batch(value: str) -> int | float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("batch must be an integer, -1, or a fraction") from exc
    return int(parsed) if parsed.is_integer() else parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO detection on VietFood67")
    parser.add_argument("--data", required=True, help="Prepared VietFood67 data YAML")
    parser.add_argument("--model", default="yolo11n.pt", help="Base .pt weights or model YAML")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch", type=parse_batch, default=-1, help="-1 enables Ultralytics AutoBatch")
    parser.add_argument("--device", default=None, help="cpu, mps, 0 or comma-separated CUDA indices")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="vietfood67_yolo11n")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--export-onnx", action="store_true")
    return parser


def resolve_save_dir(model: object, results: object) -> Path:
    """Resolve the Ultralytics run directory across single- and multi-GPU returns."""
    result_save_dir = getattr(results, "save_dir", None)
    if result_save_dir:
        return Path(result_save_dir)

    trainer = getattr(model, "trainer", None)
    trainer_save_dir = getattr(trainer, "save_dir", None)
    if trainer_save_dir:
        return Path(trainer_save_dir)

    raise RuntimeError(
        "Training finished but the output directory could not be resolved from "
        "either results.save_dir or model.trainer.save_dir."
    )


def main() -> int:
    args = build_parser().parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install training dependencies with: pip install -e '.[inference]'") from exc

    if not Path(args.data).is_file():
        raise SystemExit(f"Dataset YAML not found: {args.data}")

    model = YOLO(args.model)
    if args.resume:
        results = model.train(resume=True)
    else:
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.image_size,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            patience=args.patience,
            project=args.project,
            name=args.name,
            pretrained=True,
            plots=True,
            seed=42,
            deterministic=True,
        )
    save_dir = resolve_save_dir(model, results)
    best = save_dir / "weights" / "best.pt"
    print(f"Training output: {save_dir}")
    print(f"Best checkpoint: {best}")
    if args.export_onnx and best.is_file():
        exported = YOLO(str(best)).export(format="onnx", dynamic=True, simplify=True)
        print(f"Exported ONNX: {exported}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
