"""Phase 4: Engineering validation and correction rules for extracted values."""

from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "fields.yaml"


def load_validation_rules(config_path: Path = CONFIG_PATH) -> dict:
    """Load validation rules from fields.yaml."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config.get("fields", {})


def try_parse_numeric(value: str) -> float | None:
    """Attempt to parse a string as a number. Handles European comma decimals."""
    if value is None:
        return None
    try:
        return float(value.replace(",", "."))
    except (ValueError, TypeError):
        return None


def check_range(value: float, min_val: float, max_val: float) -> bool:
    """Check if value is within the expected range."""
    return min_val <= value <= max_val


def correct_decimal(value: str, expected_decimals: int, min_val: float, max_val: float) -> tuple[str, bool, str]:
    """Try to fix a value that's outside range by inserting/moving a decimal point.

    For example, if expected range is 4.0-10.0 and we get "618", try "6.18" or "61.8".

    Returns:
        Tuple of (corrected_value, was_corrected, reason).
    """
    num = try_parse_numeric(value)
    if num is None:
        return value, False, ""

    # Already in range
    if check_range(num, min_val, max_val):
        return value, False, ""

    # Try inserting decimal point at different positions
    digits = value.replace(".", "").replace("-", "")
    is_negative = value.startswith("-")
    prefix = "-" if is_negative else ""

    candidates = []
    for pos in range(1, len(digits)):
        candidate = f"{prefix}{digits[:pos]}.{digits[pos:]}"
        cand_num = try_parse_numeric(candidate)
        if cand_num is not None and check_range(cand_num, min_val, max_val):
            candidates.append(candidate)

    if len(candidates) == 1:
        return candidates[0], True, f"Decimal correction: {value} → {candidates[0]} (in range [{min_val}, {max_val}])"
    elif len(candidates) > 1:
        # Pick the one with expected decimal places
        for c in candidates:
            decimal_part = c.split(".")[-1] if "." in c else ""
            if len(decimal_part) == expected_decimals:
                return c, True, f"Decimal correction: {value} → {c} (matched expected {expected_decimals} decimals)"
        # Default to first candidate
        return candidates[0], True, f"Decimal correction: {value} → {candidates[0]} (best guess from {len(candidates)} candidates)"

    # Value is 10x outside range — try dividing/multiplying by 10
    if num != 0:
        for factor_label, factor in [("÷10", num / 10), ("÷100", num / 100), ("×10", num * 10)]:
            if check_range(factor, min_val, max_val):
                corrected = f"{factor:.{expected_decimals}f}"
                return corrected, True, f"Scale correction: {value} {factor_label} → {corrected}"

    return value, False, ""


def validate_field(field_name: str, value: str, rules: dict) -> dict:
    """Validate and optionally correct a single field value.

    Args:
        field_name: The field identifier.
        value: The raw recognized text.
        rules: Field definitions dict from config.

    Returns:
        Dict with keys:
            - original: The raw input value
            - corrected: The corrected value (same as original if no correction)
            - was_corrected: Whether a correction was applied
            - correction_reason: Explanation of the correction
            - in_range: Whether the final value is within the expected range
            - validation_error: Error message if validation failed, empty if ok
    """
    result = {
        "original": value,
        "corrected": value,
        "was_corrected": False,
        "correction_reason": "",
        "in_range": True,
        "validation_error": "",
    }

    field_config = rules.get(field_name)
    if field_config is None:
        result["validation_error"] = f"Unknown field: {field_name}"
        return result

    validation = field_config.get("validation", {})
    if validation.get("type") != "numeric":
        return result

    min_val = validation.get("min", float("-inf"))
    max_val = validation.get("max", float("inf"))
    expected_decimals = validation.get("expected_decimals", 2)

    num = try_parse_numeric(value)
    if num is None:
        result["in_range"] = False
        result["validation_error"] = f"Not a valid number: '{value}'"
        return result

    # Check if in range
    if check_range(num, min_val, max_val):
        return result

    # Try decimal/scale correction
    corrected, was_corrected, reason = correct_decimal(value, expected_decimals, min_val, max_val)
    result["corrected"] = corrected
    result["was_corrected"] = was_corrected
    result["correction_reason"] = reason

    # Check corrected value
    corrected_num = try_parse_numeric(corrected)
    if corrected_num is not None and check_range(corrected_num, min_val, max_val):
        result["in_range"] = True
    else:
        result["in_range"] = False
        result["validation_error"] = f"Out of range [{min_val}, {max_val}]: {corrected}"

    return result


def validate_all(predictions: dict[str, str], config_path: Path = CONFIG_PATH) -> dict[str, dict]:
    """Validate all predicted field values.

    Args:
        predictions: Dict mapping field_name → predicted_value.
        config_path: Path to fields.yaml.

    Returns:
        Dict mapping field_name → validation result dict.
    """
    rules = load_validation_rules(config_path)
    results = {}
    for field_name, value in predictions.items():
        results[field_name] = validate_field(field_name, value, rules)
    return results


def cross_validate(results: dict[str, dict], rules: dict) -> list[str]:
    """Cross-field consistency checks.

    Returns list of warning messages for inconsistent values.
    """
    warnings = []

    # Load loss (copper) should be greater than no-load loss (iron)
    no_load = try_parse_numeric(results.get("no_load_100_wfe_w", {}).get("corrected", ""))
    load = try_parse_numeric(results.get("load_loss_total_wcu_w", {}).get("corrected", ""))
    if no_load is not None and load is not None:
        if load <= no_load:
            warnings.append(
                f"Load loss ({load} W) should be greater than no-load loss ({no_load} W)"
            )

    # HV voltage should be greater than LV voltage
    hv = try_parse_numeric(results.get("hv_voltage_v", {}).get("corrected", ""))
    lv = try_parse_numeric(results.get("lv_voltage_v", {}).get("corrected", ""))
    if hv is not None and lv is not None:
        if hv <= lv:
            warnings.append(
                f"HV voltage ({hv} V) should be greater than LV voltage ({lv} V)"
            )

    # Guaranteed Wfe should be >= measured Wfe at 100%
    guaranteed_fe = try_parse_numeric(results.get("guaranteed_fe_loss_w", {}).get("corrected", ""))
    measured_fe = try_parse_numeric(results.get("no_load_100_wfe_w", {}).get("corrected", ""))
    if guaranteed_fe is not None and measured_fe is not None:
        if measured_fe > guaranteed_fe * 1.15:
            warnings.append(
                f"Measured iron loss ({measured_fe} W) exceeds guaranteed ({guaranteed_fe} W) by >15%"
            )

    # Guaranteed Wcu should be >= measured Wcu
    guaranteed_cu = try_parse_numeric(results.get("guaranteed_cu_loss_w", {}).get("corrected", ""))
    measured_cu = try_parse_numeric(results.get("load_loss_total_wcu_w", {}).get("corrected", ""))
    if guaranteed_cu is not None and measured_cu is not None:
        if measured_cu > guaranteed_cu * 1.15:
            warnings.append(
                f"Measured copper loss ({measured_cu} W) exceeds guaranteed ({guaranteed_cu} W) by >15%"
            )

    return warnings
