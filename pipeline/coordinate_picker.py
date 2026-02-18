"""Phase 2: Interactive coordinate picker for defining field bounding boxes.

Opens sample images with OpenCV. User clicks to define top-left and bottom-right
corners of each field. Saves coordinates to config/fields.yaml.

Supports multi-page forms: pass multiple images and use --page to filter.
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


def pick_coordinates(image_paths: dict[int, str], config_path: Path = CONFIG_PATH,
                     page_filter: int | None = None, recalibrate: bool = False):
    """Interactive tool to pick bounding boxes for each field.

    For each field defined in fields.yaml (with bbox [0,0,0,0]),
    prompts user to click top-left then bottom-right corners.
    Updates the YAML with the picked coordinates.

    Args:
        image_paths: Dict mapping page number to image file path.
        config_path: Path to fields.yaml.
        page_filter: If set, only calibrate fields on this page.
    """
    global _current_image, _display_image, _clicks

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load all page images
    images = {}
    for page_num, img_path in image_paths.items():
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not load image for page {page_num}: {img_path}")
            sys.exit(1)
        images[page_num] = img
        print(f"Page {page_num} loaded: {img_path} ({img.shape[1]}x{img.shape[0]})")

    fields = config.get("fields", {})

    # Filter fields by page, optionally only uncalibrated
    to_calibrate = {}
    for k, v in fields.items():
        field_page = v.get("page", 0)
        if page_filter is not None and field_page != page_filter:
            continue
        if field_page not in images:
            print(f"  Skipping {k}: page {field_page} image not provided")
            continue
        if not recalibrate and v.get("bbox") != [0, 0, 0, 0]:
            continue
        to_calibrate[k] = v

    if not to_calibrate:
        msg = "No fields to calibrate."
        if not recalibrate:
            msg += " Use --recalibrate to re-pick existing bounding boxes."
        print(msg)
        return

    total = len(to_calibrate)
    mode = "RECALIBRATE" if recalibrate else "CALIBRATE"
    print(f"\n[{mode}] Fields to calibrate: {total}")
    print("For each field, click TOP-LEFT then BOTTOM-RIGHT corner.")
    print("Press 'r' to redo current field, 's' to skip (keep current), 'q' to quit and save.\n")

    cv2.namedWindow("Coordinate Picker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Coordinate Picker", 1400, 1000)
    cv2.setMouseCallback("Coordinate Picker", _mouse_callback)

    for idx, (field_name, field_data) in enumerate(to_calibrate.items(), 1):
        field_page = field_data.get("page", 0)
        _current_image = images[field_page]
        _clicks.clear()
        _display_image = _current_image.copy()

        # Show existing bbox if recalibrating
        existing_bbox = field_data.get("bbox", [0, 0, 0, 0])
        has_existing = existing_bbox != [0, 0, 0, 0]
        if has_existing:
            ex1, ey1, ex2, ey2 = existing_bbox
            cv2.rectangle(_display_image, (ex1, ey1), (ex2, ey2), (0, 255, 0), 2)
            cv2.putText(_display_image, "CURRENT", (ex1, ey1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw label at top with progress counter
        status = "s=keep" if has_existing else "NEW"
        label = f"[{idx}/{total}] Select: {field_name} (page {field_page}) [{status}]"
        cv2.putText(_display_image, label, (10, 30),
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
            if key == ord("s"):
                print(f"  Skipped: {field_name}")
                break
            if key == ord("r"):
                _clicks.clear()
                _display_image = _current_image.copy()
                cv2.putText(_display_image, f"[{idx}/{total}] Select: {field_name} (REDO)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.imshow("Coordinate Picker", _display_image)
                continue
            if len(_clicks) >= 2:
                x1, y1 = _clicks[0]
                x2, y2 = _clicks[1]
                # Normalize: ensure top-left / bottom-right order
                bbox = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
                fields[field_name]["bbox"] = bbox
                print(f"  [{idx}/{total}] {field_name}: {bbox}")
                break

    cv2.destroyAllWindows()

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"\nCoordinates saved to {config_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pick field coordinates from sample images")
    parser.add_argument("images", type=str, nargs="+",
                        help="Page images in order: page0.png page1.png ...")
    parser.add_argument("--page", type=int, default=None,
                        help="Only calibrate fields on this page number")
    parser.add_argument("--recalibrate", action="store_true",
                        help="Show all fields (not just uncalibrated) so you can re-pick bboxes")
    args = parser.parse_args()

    # Map positional images to page numbers: first arg = page 0, second = page 1, etc.
    image_map = {i: path for i, path in enumerate(args.images)}
    pick_coordinates(image_map, page_filter=args.page, recalibrate=args.recalibrate)
