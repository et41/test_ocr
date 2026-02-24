"""
Character segmentation: split a crop image into individual character images.

Strategy:
  1. Binarize (Otsu) — ink becomes white
  2. Find connected components, filter noise / border lines, merge nearby parts
  3. If component count == expected  → use those boxes  (clean case)
  4. If count != expected            → vertical projection-profile split
     Find the N-1 deepest valleys in the column ink-sum and force-cut there
  5. Extract + resize each character to CHAR_SIZE x CHAR_SIZE
"""

import cv2
import numpy as np

CHAR_SIZE = 32  # CNN input size


# ---------------------------------------------------------------------------
# Binarization
# ---------------------------------------------------------------------------

def binarize(img_gray: np.ndarray) -> np.ndarray:
    """Return binary image: ink = white (255), background = black (0)."""
    if img_gray is None or img_gray.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    _, binary = cv2.threshold(
        img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return binary


# ---------------------------------------------------------------------------
# Connected-component segmentation
# ---------------------------------------------------------------------------

def get_char_boxes(
    binary: np.ndarray, min_area: int = 12
) -> list[tuple[int, int, int, int]]:
    """
    Find bounding boxes of ink blobs; filter noise and table border lines.
    Returns (x, y, w, h) list sorted left to right.
    """
    h, w = binary.shape
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary)

    boxes = []
    for i in range(1, len(stats)):
        x, y, cw, ch, area = stats[i]
        if area < min_area:
            continue
        if cw <= 3 and ch > h * 0.6:   # thin vertical border
            continue
        if ch <= 2 and cw > w * 0.5:   # thin horizontal line
            continue
        boxes.append((int(x), int(y), int(cw), int(ch)))

    boxes.sort(key=lambda b: b[0])
    return boxes


def merge_nearby_boxes(
    boxes: list[tuple[int, int, int, int]], gap_thresh: int = 6
) -> list[tuple[int, int, int, int]]:
    """
    Merge horizontally overlapping or close boxes into one.
    Handles multi-part characters (e.g. comma dot + tail).
    """
    if not boxes:
        return boxes

    merged = [list(boxes[0])]
    for x, y, w, h in boxes[1:]:
        px, py, pw, ph = merged[-1]
        if x - (px + pw) <= gap_thresh:        # overlap or close gap
            new_x = px
            new_y = min(py, y)
            new_w = (x + w) - px
            new_h = max(py + ph, y + h) - new_y
            merged[-1] = [new_x, new_y, new_w, new_h]
        else:
            merged.append([x, y, w, h])

    return [tuple(b) for b in merged]


# ---------------------------------------------------------------------------
# Projection-profile splitting (fallback for touching characters)
# ---------------------------------------------------------------------------

def _smooth(arr: np.ndarray, k: int = 3) -> np.ndarray:
    kernel = np.ones(k) / k
    return np.convolve(arr, kernel, mode="same")


def projection_split(
    binary: np.ndarray, n_parts: int
) -> list[tuple[int, int, int, int]]:
    """
    Force-split a binary image into n_parts character boxes by finding the
    N-1 deepest valleys in the vertical (column) ink projection.

    Returns list of (x, y, w, h) boxes spanning the full ink height.
    """
    h, w = binary.shape
    if n_parts <= 0:
        return []
    if n_parts == 1:
        return [(0, 0, w, h)]

    col_proj = binary.sum(axis=0).astype(float)

    # Ink bounding columns
    ink_cols = np.where(col_proj > 0)[0]
    if len(ink_cols) == 0:
        return _even_split(w, h, n_parts)
    x0, x1 = int(ink_cols[0]), int(ink_cols[-1]) + 1

    # Ink bounding rows
    row_proj = binary.sum(axis=1).astype(float)
    ink_rows = np.where(row_proj > 0)[0]
    y0 = int(ink_rows[0]) if len(ink_rows) else 0
    y1 = int(ink_rows[-1]) + 1 if len(ink_rows) else h

    # Smooth the column projection inside the ink region
    roi_proj = _smooth(col_proj[x0:x1], k=3)
    roi_w = x1 - x0

    # Find local minima (candidate split positions)
    minima: list[tuple[float, int]] = []
    for i in range(1, len(roi_proj) - 1):
        if roi_proj[i] <= roi_proj[i - 1] and roi_proj[i] <= roi_proj[i + 1]:
            minima.append((float(roi_proj[i]), i + x0))

    if len(minima) >= n_parts - 1:
        # Pick the N-1 shallowest valleys (where ink is thinnest)
        minima.sort(key=lambda t: t[0])
        split_cols = sorted(t[1] for t in minima[: n_parts - 1])
    else:
        # Not enough minima — evenly divide the ink region
        split_cols = [x0 + int(roi_w * i / n_parts) for i in range(1, n_parts)]

    # Build (x, y, w, h) boxes
    boxes: list[tuple[int, int, int, int]] = []
    prev_x = 0
    for sx in split_cols + [w]:
        bw = sx - prev_x
        if bw > 0:
            boxes.append((prev_x, y0, bw, y1 - y0))
        prev_x = sx
    return boxes


def _even_split(w: int, h: int, n: int) -> list[tuple[int, int, int, int]]:
    """Divide image width evenly into n columns."""
    cw = w // n
    return [(i * cw, 0, cw, h) for i in range(n)]


# ---------------------------------------------------------------------------
# Character image extraction
# ---------------------------------------------------------------------------

def extract_char_image(
    img_gray: np.ndarray,
    box: tuple[int, int, int, int],
    pad: int = 2,
    size: int = CHAR_SIZE,
) -> np.ndarray:
    """
    Crop a character region from img_gray, resize to (size, size).
    Returns float32 (size, size): ink = 1.0, background = 0.0.
    """
    h, w = img_gray.shape
    x, y, bw, bh = box
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + bw + pad)
    y2 = min(h, y + bh + pad)

    crop = img_gray[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((size, size), dtype=np.float32)

    resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return 1.0 - resized.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Main segmentation entry point
# ---------------------------------------------------------------------------

def segment_crop(
    img_gray: np.ndarray,
    expected_count: int | None = None,
    merge_gap: int = 6,
) -> list[np.ndarray] | None:
    """
    Segment a crop into individual character images.

    Args:
        img_gray:       Grayscale uint8 crop.
        expected_count: If given, return None when final count doesn't match.
                        When None (inference mode), return best-effort result.
        merge_gap:      Max pixel gap to merge nearby blobs into one character.

    Returns:
        List of (CHAR_SIZE, CHAR_SIZE) float32 arrays, or None on count mismatch.
    """
    binary = binarize(img_gray)

    # --- Step 1: connected components ---
    boxes = get_char_boxes(binary)
    boxes = merge_nearby_boxes(boxes, gap_thresh=merge_gap)

    if expected_count is None:
        # Inference: return whatever we find
        return [extract_char_image(img_gray, b) for b in boxes] if boxes else []

    # --- Step 2: if count matches, we're done ---
    if len(boxes) == expected_count:
        return [extract_char_image(img_gray, b) for b in boxes]

    # --- Step 3: projection-profile split on the whole image ---
    proj_boxes = projection_split(binary, expected_count)
    if len(proj_boxes) == expected_count:
        return [extract_char_image(img_gray, b) for b in proj_boxes]

    # Could not produce the right count
    return None
