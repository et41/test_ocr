import os
from datetime import datetime, timezone

import cv2
import numpy as np
import pytesseract
from img2table.document import PDF
from img2table.ocr import TesseractOCR
from pdf2image import convert_from_path
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

POPPLER_PATH = r"C:\poppler\poppler-25.12.0\Library\bin"


def extract_text_from_pdf(pdf_path: str) -> str:
    """Convert PDF to images and run OCR on each page. Returns full text."""
    images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
    pages = []
    for image in images:
        text = pytesseract.image_to_string(image)
        pages.append(text)
    return "\n\n".join(pages)


def extract_structured_data(pdf_path: str) -> dict:
    """Run OCR with page-level structure. Returns dict with metadata and per-page text."""
    images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
    pages = []
    for i, image in enumerate(images, start=1):
        text = pytesseract.image_to_string(image)
        pages.append({"page": i, "text": text})
    return {
        "filename": os.path.basename(pdf_path),
        "page_count": len(images),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pages": pages,
    }


def extract_tables_from_pdf(pdf_path: str, implicit_rows: bool = True) -> dict:
    """Extract tables from PDF using img2table. Returns dict with metadata and tables per page."""
    ocr = TesseractOCR(n_threads=1, lang="eng")

    # Add poppler to PATH so img2table's internal pdf2image calls work
    os.environ["PATH"] = POPPLER_PATH + os.pathsep + os.environ.get("PATH", "")

    pdf = PDF(src=pdf_path, detect_rotation=False, pdf_text_extraction=True)
    tables_by_page = pdf.extract_tables(
        ocr=ocr,
        implicit_rows=implicit_rows,
        borderless_tables=False,
        min_confidence=50,
    )

    result_pages = []
    for page_idx, tables in tables_by_page.items():
        page_tables = []
        for table in tables:
            df = table.df
            page_tables.append({
                "headers": df.iloc[0].tolist() if len(df) > 0 else [],
                "rows": df.iloc[1:].values.tolist() if len(df) > 1 else [],
                "shape": list(df.shape),
            })
        result_pages.append({"page": page_idx, "tables_found": len(tables), "tables": page_tables})

    return {
        "filename": os.path.basename(pdf_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pages": result_pages,
    }


def preprocess_image(image: np.ndarray, for_trocr: bool = False) -> np.ndarray:
    """OpenCV preprocessing pipeline for OCR.

    Args:
        image: Input image as numpy array (BGR or grayscale).
        for_trocr: If True, return after CLAHE (no binarization).
                   If False, apply adaptive binarization + deskew for Tesseract.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    if for_trocr:
        return enhanced

    # Adaptive binarization for Tesseract
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Deskew
    coords = np.column_stack(np.where(binary < 255))
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) > 0.5:
            h, w = binary.shape
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            binary = cv2.warpAffine(binary, matrix, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)

    return binary


def extract_handwritten_tables(pdf_path: str) -> dict:
    """Extract handwritten text from table cells using TrOCR.

    Uses img2table to detect table structure, then runs TrOCR on each cell.
    """
    MODEL_NAME = "microsoft/trocr-base-handwritten"
    print(f"Loading TrOCR model ({MODEL_NAME})...")
    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

    os.environ["PATH"] = POPPLER_PATH + os.pathsep + os.environ.get("PATH", "")

    pdf = PDF(src=pdf_path, detect_rotation=False, pdf_text_extraction=False)
    # Extract tables without OCR — we only need the bounding boxes
    ocr = TesseractOCR(n_threads=1, lang="eng")
    tables_by_page = pdf.extract_tables(
        ocr=ocr,
        implicit_rows=True,
        borderless_tables=False,
        min_confidence=50,
    )

    images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH, dpi=300)

    result_pages = []
    for page_idx, tables in tables_by_page.items():
        if page_idx >= len(images):
            continue
        page_image = np.array(images[page_idx])

        page_tables = []
        for table in tables:
            table_rows = []
            for row_idx, cells in table.content.items():
                row_cells = []
                for cell in cells:
                    x1, y1 = cell.bbox.x1, cell.bbox.y1
                    x2, y2 = cell.bbox.x2, cell.bbox.y2
                    cell_img = page_image[y1:y2, x1:x2]
                    if cell_img.size == 0:
                        text = ""
                    else:
                        preprocessed = preprocess_image(cell_img, for_trocr=True)
                        cell_pil = Image.fromarray(preprocessed).convert("RGB")
                        pixel_values = processor(images=cell_pil, return_tensors="pt").pixel_values
                        generated_ids = model.generate(pixel_values)
                        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                    row_cells.append(text)
                table_rows.append(row_cells)

            page_tables.append({
                "rows": table_rows,
                "shape": [len(table_rows), max((len(r) for r in table_rows), default=0)],
            })

        result_pages.append({
            "page": page_idx,
            "tables_found": len(tables),
            "tables": page_tables,
        })

    return {
        "filename": os.path.basename(pdf_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pages": result_pages,
    }
