import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch


API_DIR = Path(__file__).resolve().parents[1]
MAIN_PATH = API_DIR / "main.py"


def load_main_with_stubbed_dependencies():
    cv2_stub = ModuleType("cv2")
    numpy_stub = ModuleType("numpy")
    ultralytics_stub = ModuleType("ultralytics")
    ultralytics_stub.YOLO = Mock()
    detection_code_stub = ModuleType("detection_code")
    detection_code_stub.Face = Mock()

    dependencies = {
        "cv2": cv2_stub,
        "numpy": numpy_stub,
        "ultralytics": ultralytics_stub,
        "detection_code": detection_code_stub,
    }
    spec = importlib.util.spec_from_file_location("tefillin_main_under_test", MAIN_PATH)
    module = importlib.util.module_from_spec(spec)

    with patch.dict(sys.modules, dependencies):
        spec.loader.exec_module(module)

    return module


class TefillinPlacementTests(unittest.TestCase):
    def setUp(self):
        self.main = load_main_with_stubbed_dependencies()
        self.main.initialized = True
        self.main.detect_eyes = Mock(return_value=[[0, 40, 20], [0, 60, 20]])
        self.main.calc_hairline = Mock(return_value=100)
        self.main.detect_tefillin = Mock(
            return_value=[[45, 80], [45, 90], [55, 80], [55, 90]]
        )

    def check_placement(self):
        return self.main.tef_good(object())

    def test_requires_reference_image(self):
        self.main.initialized = False

        self.assertEqual(self.check_placement(), [False, "Need reference image"])

    def test_accepts_tefillin_between_eyes_and_above_hairline(self):
        self.assertEqual(self.check_placement(), [True, ""])

    def test_rejects_missing_eyes(self):
        self.main.detect_eyes.return_value = None

        self.assertEqual(self.check_placement(), [False, "No eyes detected"])

    def test_returns_hairline_error(self):
        self.main.calc_hairline.return_value = "Can't detect chin + nose"

        self.assertEqual(
            self.check_placement(), [False, "Can't detect chin + nose"]
        )

    def test_rejects_missing_tefillin(self):
        self.main.detect_tefillin.return_value = []

        self.assertEqual(self.check_placement(), [False, "No tefillin detected"])

    def test_rejects_tefillin_left_of_eyes(self):
        self.main.detect_tefillin.return_value = [
            [25, 80], [25, 90], [35, 80], [35, 90]
        ]

        self.assertEqual(self.check_placement(), [False, "Too far left"])

    def test_rejects_tefillin_right_of_eyes(self):
        self.main.detect_tefillin.return_value = [
            [65, 80], [65, 90], [75, 80], [75, 90]
        ]

        self.assertEqual(self.check_placement(), [False, "Too far right"])

    def test_rejects_tefillin_below_hairline(self):
        self.main.detect_tefillin.return_value = [
            [45, 105], [45, 110], [55, 105], [55, 110]
        ]

        self.assertEqual(self.check_placement(), [False, "Too low"])

    def test_compares_head_pose_and_returns_adjustment_guidance(self):
        guidance = self.main.compare_poses(
            current_pose=[[10], [5], [-6]],
            reference_pose=[[0], [0], [0]],
        )

        self.assertEqual(
            guidance,
            [
                ("Turn right", 5),
                ("Look down", 10),
                ("Tilt head right", 6),
            ],
        )


if __name__ == "__main__":
    unittest.main()
