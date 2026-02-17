"""Phase 5: CLI review and correction tool for extracted values."""

import csv
import json
from pathlib import Path

LABELS_CSV = Path(__file__).resolve().parent.parent / "data" / "labels.csv"


def review_results(results: dict[str, dict], predictions: dict[str, tuple[str, float]],
                   confidence_threshold: float = 0.85) -> dict[str, dict]:
    """Interactive CLI review of extracted and validated values.

    Shows all fields, flags those with low confidence or failed validation.
    User can accept or correct each flagged value.

    Args:
        results: Validation results from postprocess.rules.validate_all().
        predictions: Dict mapping field_name → (predicted_text, confidence).
        confidence_threshold: Flag fields below this confidence.

    Returns:
        Updated results dict with user corrections applied.
    """
    flagged = []
    accepted = []

    for field_name, result in results.items():
        pred_text, confidence = predictions.get(field_name, ("", 0.0))
        needs_review = (
            confidence < confidence_threshold
            or not result.get("in_range", True)
            or result.get("was_corrected", False)
        )
        if needs_review:
            flagged.append((field_name, result, confidence))
        else:
            accepted.append((field_name, result, confidence))

    # Show accepted fields
    if accepted:
        print("\n=== Accepted Fields ===")
        for field_name, result, confidence in accepted:
            value = result.get("corrected", result.get("original", ""))
            print(f"  {field_name}: {value} (confidence: {confidence:.1%})")

    # Review flagged fields
    if flagged:
        print(f"\n=== Flagged Fields ({len(flagged)}) ===")
        print("For each field: press Enter to accept, or type the correct value.")
        print()

        for field_name, result, confidence in flagged:
            original = result.get("original", "")
            corrected = result.get("corrected", original)
            reason = result.get("correction_reason", "")
            error = result.get("validation_error", "")

            print(f"  Field: {field_name}")
            print(f"  OCR value: {original} (confidence: {confidence:.1%})")
            if result.get("was_corrected"):
                print(f"  Auto-corrected to: {corrected}")
                print(f"  Reason: {reason}")
            if error:
                print(f"  Validation error: {error}")

            user_input = input("  Correct value [Enter=accept]: ").strip()
            if user_input:
                results[field_name]["corrected"] = user_input
                results[field_name]["was_corrected"] = True
                results[field_name]["correction_reason"] = "Manual correction by user"
                results[field_name]["in_range"] = True
                results[field_name]["validation_error"] = ""
                print(f"  → Updated to: {user_input}")
            else:
                print(f"  → Accepted: {corrected}")
            print()
    else:
        print("\nNo fields flagged for review.")

    return results


def save_corrections_for_retraining(results: dict[str, dict], crop_dir: Path,
                                    pdf_name: str, labels_csv: Path = LABELS_CSV):
    """Save user-corrected values back to labels.csv for model retraining.

    Only saves fields that were manually corrected.
    """
    corrections = {
        field_name: result["corrected"]
        for field_name, result in results.items()
        if result.get("correction_reason") == "Manual correction by user"
    }

    if not corrections:
        return

    # Load existing labels to avoid duplicates
    existing = set()
    if labels_csv.exists():
        with open(labels_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing.add((row["image_path"], row["field_name"]))

    write_header = not labels_csv.exists()
    with open(labels_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["image_path", "field_name", "value"])

        for field_name, value in corrections.items():
            crop_path = crop_dir / f"{pdf_name}_{field_name}.png"
            key = (str(crop_path), field_name)
            if key not in existing:
                writer.writerow([str(crop_path), field_name, value])

    print(f"Saved {len(corrections)} correction(s) to {labels_csv} for retraining.")


def format_output(results: dict[str, dict], predictions: dict[str, tuple[str, float]],
                  warnings: list[str] = None) -> dict:
    """Format results as a structured JSON-serializable dict.

    Args:
        results: Validation results.
        predictions: Field → (text, confidence) mapping.
        warnings: Cross-validation warnings.

    Returns:
        Structured output dict.
    """
    fields = {}
    for field_name, result in results.items():
        _, confidence = predictions.get(field_name, ("", 0.0))
        fields[field_name] = {
            "value": result.get("corrected", result.get("original", "")),
            "confidence": round(confidence, 4),
            "in_range": result.get("in_range", True),
            "was_corrected": result.get("was_corrected", False),
            "correction_reason": result.get("correction_reason", ""),
        }

    output = {
        "fields": fields,
        "summary": {
            "total_fields": len(fields),
            "fields_in_range": sum(1 for f in fields.values() if f["in_range"]),
            "fields_corrected": sum(1 for f in fields.values() if f["was_corrected"]),
            "avg_confidence": round(
                sum(f["confidence"] for f in fields.values()) / max(len(fields), 1), 4
            ),
        },
    }

    if warnings:
        output["warnings"] = warnings

    return output
