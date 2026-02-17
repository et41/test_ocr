"""Phase 2: Interactive coordinate picker for defining field bounding boxes.

Opens a sample image with OpenCV. User clicks to define top-left and bottom-right
corners of each field. Saves coordinates to config/fields.yaml.
"""

import sys
from pathlib import Path

import cv2
import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "fields.yaml"

# State for mouse callback
_clicks: list[tuple[int, int]] = []
_current_image = None
_display_image = None


def _mouse_callback(event, x, y, flags, param):
    global _display_image
    if event == cv2.EVENT_LBUTTONDOWN:
        _clicks.append((x, y))
        # Draw marker
        cv2.circle(_display_image, (x, y), 5, (0, 0, 255), -1)
        if len(_clicks) % 2 == 0:
            # Draw rectangle for completed pair
            pt1 = _clicks[-2]
            pt2 = _clicks[-1]
            cv2.rectangle(_display_image, pt1, pt2, (0, 255, 0), 2)
        cv2.imshow("Coordinate Picker", _display_image)


def pick_coordinates(image_path: str, config_path: Path = CONFIG_PATH):
    """Interactive tool to pick bounding boxes for each field.

    For each field defined in fields.yaml (with bbox [0,0,0,0]),
    prompts user to click top-left then bottom-right corners.
    Updates the YAML with the picked coordinates.
    """
    global _current_image, _display_image, _clicks

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    _current_image = cv2.imread(image_path)
    if _current_image is None:
        print(f"Error: Could not load image: {image_path}")
        sys.exit(1)

    fields = config.get("fields", {})
    uncalibrated = {k: v for k, v in fields.items() if v.get("bbox") == [0, 0, 0, 0]}

    if not uncalibrated:
        print("All fields already have coordinates. To recalibrate, reset bbox to [0, 0, 0, 0] in fields.yaml.")
        return

    print(f"Image loaded: {image_path}")
    print(f"Fields to calibrate: {len(uncalibrated)}")
    print("For each field, click TOP-LEFT then BOTTOM-RIGHT corner.")
    print("Press 'r' to redo current field, 'q' to quit and save.\n")

    cv2.namedWindow("Coordinate Picker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Coordinate Picker", 1400, 1000)
    cv2.setMouseCallback("Coordinate Picker", _mouse_callback)

    for field_name, field_data in uncalibrated.items():
        _clicks.clear()
        _display_image = _current_image.copy()

        # Draw label at top
        cv2.putText(_display_image, f"Select: {field_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(_display_image, field_data.get("description", ""), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)
        cv2.imshow("Coordinate Picker", _display_image)

        while True:
            key = cv2.waitKey(50) & 0xFF
            if key == ord("q"):
                print("Quitting. Saving progress...")
                with open(config_path, "w") as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                cv2.destroyAllWindows()
                return
            if key == ord("r"):
                _clicks.clear()
                _display_image = _current_image.copy()
                cv2.putText(_display_image, f"Select: {field_name} (REDO)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.imshow("Coordinate Picker", _display_image)
                continue
            if len(_clicks) >= 2:
                break

        x1, y1 = _clicks[0]
        x2, y2 = _clicks[1]
        # Normalize: ensure top-left / bottom-right order
        bbox = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        fields[field_name]["bbox"] = bbox
        print(f"  {field_name}: {bbox}")

    cv2.destroyAllWindows()

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"\nCoordinates saved to {config_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pick field coordinates from a sample image")
    parser.add_argument("image", type=str, help="Path to a sample page image (300 DPI PNG)")
    args = parser.parse_args()

    pick_coordinates(args.image)
