"""CalcuCalo image-analysis baseline."""

from .analyzer import FoodImageAnalyzer
from .detector import OnnxYoloDetector, UltralyticsDetector
from .portion import PortionEstimator

__all__ = [
    "FoodImageAnalyzer",
    "OnnxYoloDetector",
    "PortionEstimator",
    "UltralyticsDetector",
]
__version__ = "0.1.0"
