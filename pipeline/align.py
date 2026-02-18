"""Form alignment using multi-anchor template matching.

Finds the translation offset between a scanned form and the reference image
by matching distinctive structural regions (headers, grid corners, logos).
"""

import cv2
import numpy as np


def _extract_anchor_regions(image: np.ndarray) -> list[tuple[np.ndarray, int, int]]:
    """Extract multiple anchor patches from known structural positions.

    Returns list of (patch, center_x, center_y) tuples.
    We use regions that contain printed form structure (not handwritten areas).
    """
    h, w = image.shape[:2]
    anchors = []

    # Top-left: ENERGOIN logo area
    anchors.append((image[10:120, 10:350], 180, 65))
    # Top-right: TURKAK Report No area
    anchors.append((image[10:80, w - 500:w - 50], w - 275, 45))
    # Form title: "TRANSFORMER TEST RAW DATA FORM"
    anchors.append((image[80:150, 300:1200], 750, 115))
    # Bottom disclaimer text
    anchors.append((image[h - 120:h - 50, 50:1200], 625, h - 85))

    return anchors


def compute_offset(ref_image: np.ndarray, target_image: np.ndarray,
                   search_margin: int = 150) -> tuple[int, int]:
    """Compute (dx, dy) translation offset to align target to reference.

    Uses template matching on multiple anchor regions and takes the median offset.

    Args:
        ref_image: Reference image (the one used for coordinate calibration).
        target_image: Image to align.
        search_margin: How many pixels to search around each anchor position.

    Returns:
        (dx, dy) offset to apply to target image coordinates.
        Positive dx means target content is shifted right relative to reference.
    """
    ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY) if len(ref_image.shape) == 3 else ref_image
    tgt_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY) if len(target_image.shape) == 3 else target_image

    anchors = _extract_anchor_regions(ref_gray)
    offsets_x = []
    offsets_y = []

    for patch, cx, cy in anchors:
        ph, pw = patch.shape[:2]
        if ph < 10 or pw < 10:
            continue

        # Define search region in target image
        sx1 = max(0, cx - pw // 2 - search_margin)
        sy1 = max(0, cy - ph // 2 - search_margin)
        sx2 = min(tgt_gray.shape[1], cx + pw // 2 + search_margin)
        sy2 = min(tgt_gray.shape[0], cy + ph // 2 + search_margin)

        search_region = tgt_gray[sy1:sy2, sx1:sx2]
        if search_region.shape[0] < ph or search_region.shape[1] < pw:
            continue

        result = cv2.matchTemplate(search_region, patch, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val < 0.3:
            continue

        # Found location in search region
        found_x = sx1 + max_loc[0] + pw // 2
        found_y = sy1 + max_loc[1] + ph // 2

        dx = found_x - cx
        dy = found_y - cy
        offsets_x.append(dx)
        offsets_y.append(dy)

    if not offsets_x:
        return 0, 0

    # Use median to be robust to outliers
    dx = int(np.median(offsets_x))
    dy = int(np.median(offsets_y))
    return dx, dy


def align_image(ref_image: np.ndarray, target_image: np.ndarray) -> np.ndarray:
    """Align target image to reference using translation offset.

    Returns shifted target image with same dimensions.
    """
    dx, dy = compute_offset(ref_image, target_image)

    if dx == 0 and dy == 0:
        return target_image

    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    h, w = ref_image.shape[:2]
    aligned = cv2.warpAffine(target_image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return aligned
