from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATHS = (
    ROOT / "API" / "hairline_detection" / "hairlineAI.pt",
    ROOT / "API" / "tefillin_detection" / "runs" / "detect" / "train6" / "weights" / "best.pt",
)


def main():
    for model_path in MODEL_PATHS:
        if not model_path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        model = YOLO(str(model_path))
        exported_path = model.export(
            format="coreml",
            imgsz=640,
            nms=model.task == "detect",
        )
        expected_path = model_path.with_suffix(".mlpackage")
        if Path(exported_path).resolve() != expected_path.resolve():
            raise RuntimeError(
                f"Expected Core ML export at {expected_path}, got {exported_path}"
            )
        print(f"Exported {model_path.name} to {expected_path}")


if __name__ == "__main__":
    main()
