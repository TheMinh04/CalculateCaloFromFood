from __future__ import annotations

from .calibration import PlateScaleEstimator, manual_scale
from .detector import Detector
from .domain import AnalysisItem, AnalysisResult, ScaleCalibration
from .image_io import ImageInput, load_rgb_image
from .masks import bbox_mask, mask_quality, mask_to_polygons, normalize_mask
from .portion import PortionEstimator
from .segmenter import Segmenter


class FoodImageAnalyzer:
    def __init__(
        self,
        detector: Detector,
        segmenter: Segmenter,
        portion_estimator: PortionEstimator,
        exclude_class_ids: set[int] | None = None,
    ) -> None:
        self.detector = detector
        self.segmenter = segmenter
        self.portion_estimator = portion_estimator
        self.exclude_class_ids = exclude_class_ids if exclude_class_ids is not None else {27}
        self.plate_scale_estimator = PlateScaleEstimator()

    def analyze(
        self,
        source: ImageInput,
        *,
        plate_diameter_cm: float | None = None,
        cm_per_pixel: float | None = None,
    ) -> AnalysisResult:
        image = load_rgb_image(source)
        height, width = image.shape[:2]
        detections = [
            detection
            for detection in self.detector.predict(image)
            if detection.class_id not in self.exclude_class_ids
        ]

        calibration: ScaleCalibration | None = None
        warnings: list[str] = []
        if cm_per_pixel is not None:
            calibration = manual_scale(cm_per_pixel)
        elif plate_diameter_cm is not None:
            calibration = self.plate_scale_estimator.estimate(image, plate_diameter_cm)
            if calibration is None:
                warnings.append(
                    "Khong tim thay duong tron cua dia/bat; khoi luong dang dung serving prior."
                )
        else:
            warnings.append(
                "Anh khong co ty le met; khoi luong chi la serving prior. "
                "Gui cm_per_pixel hoac plate_diameter_cm de uoc luong hinh hoc."
            )

        missing = [detection for detection in detections if detection.mask is None]
        generated_masks = iter(self.segmenter.segment(image, missing)) if missing else iter(())

        items: list[AnalysisItem] = []
        for detection in detections:
            if detection.mask is None:
                try:
                    mask = next(generated_masks)
                except StopIteration:
                    mask = bbox_mask((height, width), detection.bbox)
            else:
                mask = detection.mask
            mask = normalize_mask(mask, (height, width))
            quality = mask_quality(mask, detection.bbox)
            portion = self.portion_estimator.estimate(
                detection.label,
                mask,
                calibration,
                mask_confidence=quality,
            )
            items.append(
                AnalysisItem(
                    class_id=detection.class_id,
                    label=detection.label,
                    detection_confidence=detection.confidence,
                    bbox=detection.bbox,
                    polygons=mask_to_polygons(mask),
                    portion=portion,
                )
            )

        if not items:
            warnings.append("Khong phat hien mon an nao vuot nguong confidence.")
        return AnalysisResult(
            image_width=width,
            image_height=height,
            items=items,
            calibration=calibration,
            warnings=warnings,
        )

