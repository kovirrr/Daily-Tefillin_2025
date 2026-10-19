from pathlib import Path
import argparse
import sys

import cv2
from ultralytics import YOLO


API_DIR = Path(__file__).resolve().parent
ROOT = API_DIR.parent
DEFAULT_REFERENCE = API_DIR / "people" / "kovi.JPG"
HAIRLINE_MODEL_PATH = API_DIR / "hairline_detection" / "hairlineAI.pt"
TEFILLIN_MODEL_PATH = API_DIR / "tefillin_detection" / "runs" / "detect" / "train6" / "weights" / "best.pt"
POSE_GUIDANCE_THRESHOLD_DEGREES = 8.0

if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

import main as tefillin_main


hairline_model = YOLO(str(HAIRLINE_MODEL_PATH))
tefillin_model = YOLO(str(TEFILLIN_MODEL_PATH))


def detect_hairline_local(image):
    try:
        results = hairline_model(image, verbose=False)
        result = results[0]
        if result.masks is None or len(result.masks.xy) == 0:
            return None

        confidences = result.boxes.conf.cpu().numpy()
        if len(confidences) == 0:
            return None
        best_index = int(confidences.argmax())
        return result.masks.xy[best_index].tolist()
    except Exception:
        return None


def detect_tefillin_local(image):
    try:
        results = tefillin_model(image, verbose=False)
        result = results[0]
        confidences = result.boxes.conf.cpu().numpy()
        if len(confidences) == 0:
            return []

        best_index = int(confidences.argmax())
        x1, y1, x2, y2 = result.boxes.xyxy[best_index].cpu().numpy()
        return [
            [x1, y1],
            [x1, y2],
            [x2, y1],
            [x2, y2],
        ]
    except Exception:
        return []


def draw_overlay(frame, hairline_y, tefillin_points, chin, nose):
    if hairline_y is not None and not isinstance(hairline_y, str):
        cv2.line(
            frame,
            (0, int(hairline_y)),
            (frame.shape[1], int(hairline_y)),
            (0, 0, 255),
            2,
        )
        cv2.putText(
            frame,
            "hairline",
            (10, max(25, int(hairline_y) - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    if tefillin_points:
        top_left = (int(tefillin_points[0][0]), int(tefillin_points[0][1]))
        bottom_right = (int(tefillin_points[3][0]), int(tefillin_points[3][1]))
        cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)
        cv2.putText(
            frame,
            "tefillin",
            (top_left[0], max(25, top_left[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )

    if chin:
        cv2.circle(frame, (int(chin[1]), int(chin[2])), 7, (255, 0, 255), -1)
        cv2.putText(
            frame,
            "chin",
            (int(chin[1]) + 10, int(chin[2])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 255),
            2,
        )

    if nose:
        cv2.circle(frame, (int(nose[1]), int(nose[2])), 7, (0, 255, 255), -1)
        cv2.putText(
            frame,
            "nose",
            (int(nose[1]) + 10, int(nose[2])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )


def draw_face_guidance(frame, guidance, aligned):
    color = (0, 200, 0) if aligned else (0, 165, 255)
    for index, text in enumerate(guidance):
        cv2.putText(
            frame,
            f"FACE: {text}",
            (10, 190 + index * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
        )


def draw_check_result(frame, passed):
    color = (0, 180, 0) if passed else (0, 0, 220)
    border = max(8, frame.shape[1] // 80)
    cv2.rectangle(
        frame,
        (0, 0),
        (frame.shape[1] - 1, frame.shape[0] - 1),
        color,
        border,
    )
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 48), color, -1)
    cv2.putText(
        frame,
        "PLACEMENT CHECK: GOOD" if passed else "PLACEMENT CHECK: NOT GOOD",
        (12, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Live tefillin test with hairline, tefillin, chin, and nose highlights.",
    )
    parser.add_argument(
        "--reference",
        default=str(DEFAULT_REFERENCE),
        help="Path to the reference image used to initialize the live hairline calculation.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index for cv2.VideoCapture.",
    )
    return parser.parse_args()


def safe_face_features(frame):
    try:
        chin, nose = tefillin_main.face_features(frame)
        return chin, nose
    except Exception:
        return None, None


def safe_face_guidance(frame, reference_ready, reference_message):
    if not reference_ready:
        return [f"alignment unavailable ({reference_message})"], False
    try:
        current_pose = tefillin_main.detect_head_pose(frame)
        adjustments = tefillin_main.compare_poses(
            current_pose,
            tefillin_main.ref_pos,
            threshold=POSE_GUIDANCE_THRESHOLD_DEGREES,
        )
    except Exception:
        return ["face not detected; center your face in the camera"], False

    if not adjustments:
        return [f"aligned (within {POSE_GUIDANCE_THRESHOLD_DEGREES:g} degrees of reference)"], True
    return [f"{instruction} ({degrees:.1f} degrees)" for instruction, degrees in adjustments], False


def safe_calc_hairline(frame, reference_ready):
    if not reference_ready:
        return None, "reference not initialized"
    try:
        hairline_y = tefillin_main.calc_hairline(frame)
        if isinstance(hairline_y, str):
            return None, hairline_y
        return hairline_y, None
    except Exception as exc:
        return None, f"hairline error: {exc}"


def safe_detect_tefillin(frame):
    try:
        tefillin_points = tefillin_main.detect_tefillin(frame)
        if not tefillin_points:
            return [], "tefillin missing"
        return tefillin_points, None
    except Exception as exc:
        return [], f"tefillin error: {exc}"


def build_status_lines(reference_ready, reference_message, hairline_y, hairline_message, tefillin_points, tefillin_message, chin, nose):
    return [
        f"reference: {'ok' if reference_ready else reference_message}",
        f"hairline: {'ok' if hairline_y is not None else (hairline_message or 'missing')}",
        f"tefillin: {'ok' if tefillin_points else (tefillin_message or 'missing')}",
        f"chin: {'ok' if chin else 'missing'}",
        f"nose: {'ok' if nose else 'missing'}",
    ]


def main():
    args = parse_args()
    reference_path = Path(args.reference).expanduser().resolve()

    if not reference_path.exists():
        raise FileNotFoundError(f"Reference image not found: {reference_path}")
    if not HAIRLINE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Hairline model not found: {HAIRLINE_MODEL_PATH}")
    if not TEFILLIN_MODEL_PATH.exists():
        raise FileNotFoundError(f"Tefillin model not found: {TEFILLIN_MODEL_PATH}")

    tefillin_main.detect_hairline = detect_hairline_local
    tefillin_main.detect_tefillin = detect_tefillin_local
    tefillin_main.initialized = False
    tefillin_main.send_warning = lambda warning: None

    reference_image = tefillin_main.read(str(reference_path))
    if reference_image is None:
        raise ValueError(f"Could not read reference image: {reference_path}")

    reference_ready = False
    reference_message = "reference init failed"
    try:
        ref_eyes = tefillin_main.detect_eyes(reference_image)
        ref_hairline = tefillin_main.detect_hairline(reference_image)
        ref_chin, ref_nose = safe_face_features(reference_image)
        if not ref_eyes:
            reference_message = "reference eyes missing"
        elif not ref_hairline:
            reference_message = "reference hairline missing"
        elif not ref_chin or not ref_nose:
            reference_message = "reference chin/nose missing"
        else:
            tefillin_main.initialize(reference_image)
            if hasattr(tefillin_main, "ref_pos") and hasattr(tefillin_main, "ref_top_dis") and hasattr(tefillin_main, "ref_bottom_dis"):
                reference_ready = True
                reference_message = "ok"
    except Exception as exc:
        reference_message = f"reference error: {exc}"

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break

            capture_frame = frame.copy()
            chin, nose = safe_face_features(frame)
            face_guidance, face_aligned = safe_face_guidance(
                frame,
                reference_ready,
                reference_message,
            )
            hairline_y, hairline_message = safe_calc_hairline(frame, reference_ready)
            tefillin_points, tefillin_message = safe_detect_tefillin(frame)

            draw_overlay(frame, hairline_y, tefillin_points, chin, nose)

            checks = build_status_lines(
                reference_ready,
                reference_message,
                hairline_y,
                hairline_message,
                tefillin_points,
                tefillin_message,
                chin,
                nose,
            )

            for index, text in enumerate(checks):
                cv2.putText(
                    frame,
                    text,
                    (10, 30 + index * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

            draw_face_guidance(frame, face_guidance, face_aligned)

            cv2.putText(
                frame,
                "Press w to check placement | q to quit",
                (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 200),
                2,
            )

            cv2.imshow("Live Tefillin Test", frame)
            key = cv2.waitKey(1) & 0xFF
            try:
                live_window_open = (
                    cv2.getWindowProperty("Live Tefillin Test", cv2.WND_PROP_VISIBLE) >= 1
                )
            except cv2.error:
                live_window_open = False
            if not live_window_open:
                break
            if key == ord("q"):
                break
            if key == ord("w"):
                reasons = []
                try:
                    placement_ok, placement_reason = tefillin_main.tef_good(capture_frame)
                    if not placement_ok:
                        reasons.append(placement_reason or "Placement checker rejected the image.")
                except Exception as exc:
                    placement_ok = False
                    reasons.append(f"Placement checker error: {exc}")

                check_passed = placement_ok
                if not face_aligned:
                    print("Face pose guidance only (does not affect placement result): " + "; ".join(face_guidance))

                if check_passed:
                    print("Tefillin placement check: GOOD")
                else:
                    print("Tefillin placement check: NOT GOOD")
                    for reason in reasons:
                        print(f"- {reason}")

                checked_frame = capture_frame.copy()
                draw_overlay(checked_frame, hairline_y, tefillin_points, chin, nose)
                draw_check_result(checked_frame, check_passed)
                cv2.imshow("Tefillin Check", checked_frame)
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
