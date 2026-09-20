import numpy as np

from calcucalo.detector import OnnxYoloDetector


def test_decode_yolov10_end_to_end_output() -> None:
    detector = OnnxYoloDetector.__new__(OnnxYoloDetector)
    detector.confidence = 0.25
    detector.iou = 0.6
    detector.names = {0: "Com", 1: "Pho"}
    output = np.array(
        [[[10, 20, 30, 40, 0.9, 1], [5, 5, 15, 15, 0.1, 0]]],
        dtype=np.float32,
    )

    boxes, scores, class_ids = detector._decode(output)

    assert boxes.tolist() == [[10.0, 20.0, 30.0, 40.0]]
    assert scores.tolist() == [np.float32(0.9)]
    assert class_ids.tolist() == [1]
