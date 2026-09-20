from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import UnidentifiedImageError

from .analyzer import FoodImageAnalyzer
from .detector import create_detector
from .portion import PortionEstimator
from .segmenter import create_segmenter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
app = FastAPI(title="CalcuCalo Vision API", version="0.1.0")


@lru_cache(maxsize=1)
def get_analyzer() -> FoodImageAnalyzer:
    model_path = os.getenv("CALCUCALO_MODEL")
    if not model_path:
        raise RuntimeError("Set CALCUCALO_MODEL to a trained YOLO .pt or .onnx file")
    device = os.getenv("CALCUCALO_DEVICE")
    segmenter_name = os.getenv("CALCUCALO_SEGMENTER", "grabcut")
    sam_model = os.getenv("CALCUCALO_SAM_MODEL", "sam2.1_t.pt")
    priors = os.getenv(
        "CALCUCALO_PORTION_PRIORS",
        str(PROJECT_ROOT / "configs" / "portion_priors.yaml"),
    )
    detector = create_detector(
        model_path,
        classes_path=PROJECT_ROOT / "configs" / "vietfood67_classes.yaml",
        device=device,
    )
    segmenter = create_segmenter(segmenter_name, sam_model=sam_model, device=device)
    return FoodImageAnalyzer(detector, segmenter, PortionEstimator(priors))


@app.get("/health")
def health() -> dict[str, object]:
    model_path = os.getenv("CALCUCALO_MODEL")
    return {
        "status": "ok",
        "model_configured": bool(model_path),
        "model_exists": bool(model_path and Path(model_path).is_file()),
    }


@app.post("/api/v1/food/analyze")
async def analyze_food(
    image: Annotated[UploadFile, File()],
    plate_diameter_cm: Annotated[float | None, Form()] = None,
    cm_per_pixel: Annotated[float | None, Form()] = None,
) -> dict[str, object]:
    if plate_diameter_cm is not None and cm_per_pixel is not None:
        raise HTTPException(status_code=422, detail="Use only one scale calibration method")
    payload = await image.read()
    if not payload:
        raise HTTPException(status_code=422, detail="Empty image")
    if len(payload) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image exceeds 15 MB")
    try:
        analyzer = get_analyzer()
        result = analyzer.analyze(
            payload,
            plate_diameter_cm=plate_diameter_cm,
            cm_per_pixel=cm_per_pixel,
        )
    except (UnidentifiedImageError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return result.to_dict()
