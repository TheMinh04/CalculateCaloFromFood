from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import UnidentifiedImageError

from .analyzer import FoodImageAnalyzer
from .detector import create_detector
from .nutrition import NutritionCatalog
from .portion import PortionEstimator
from .segmenter import create_segmenter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
app = FastAPI(title="CalcuCalo Vision API", version="0.3.0")


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
    component_detector = detector
    component_model = os.getenv("CALCUCALO_COMPONENT_MODEL")
    if component_model:
        component_detector = create_detector(
            component_model,
            classes_path=PROJECT_ROOT / "configs" / "vietfood67_classes.yaml",
            device=device,
        )
    segmenter = create_segmenter(segmenter_name, sam_model=sam_model, device=device)
    catalog = os.getenv(
        "CALCUCALO_FOOD_CATALOG",
        str(PROJECT_ROOT / "configs" / "food_catalog.json"),
    )
    return FoodImageAnalyzer(
        detector,
        segmenter,
        PortionEstimator(priors),
        nutrition_catalog=NutritionCatalog(catalog),
        component_detector=component_detector,
        enable_component_pass=os.getenv("CALCUCALO_COMPONENT_PASS", "false").casefold()
        in {"1", "true", "yes"},
    )


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
    response_format: Annotated[str, Form()] = "full",
    component_overrides_json: Annotated[str | None, Form()] = None,
) -> dict[str, object]:
    if plate_diameter_cm is not None and cm_per_pixel is not None:
        raise HTTPException(status_code=422, detail="Use only one scale calibration method")
    if response_format not in {"full", "nutrition"}:
        raise HTTPException(status_code=422, detail="response_format must be full or nutrition")
    component_overrides = None
    if component_overrides_json:
        try:
            component_overrides = json.loads(component_overrides_json)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=422, detail="Invalid component_overrides_json") from exc
        if not isinstance(component_overrides, dict):
            raise HTTPException(status_code=422, detail="component_overrides_json must be an object")
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
            component_overrides=component_overrides,
        )
    except (UnidentifiedImageError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if response_format == "nutrition":
        return result.to_nutrition_dict(compact=True, unwrap_single=True)
    return result.to_dict()
