"""Shared scientific-quality policy for catalyst-screen results."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

WEAK_MINIMUM_IMAG_THRESHOLD_CM1 = 50.0


def minimum_vibration_status(frequencies: Iterable[float]) -> dict[str, Any]:
    """Classify the vibration spectrum of a calculated minimum."""
    negative = [float(frequency) for frequency in frequencies if float(frequency) < 0.0]
    weak = len(negative) == 1 and abs(negative[0]) < WEAK_MINIMUM_IMAG_THRESHOLD_CM1
    if not negative:
        status = "auto_valid"
        flags: list[str] = []
        issues: list[str] = []
    elif weak:
        status = "review"
        flags = ["weak_minimum_imag"]
        issues = []
    else:
        status = "invalid"
        flags = []
        issues = [f"expected_0_imag_found_{len(negative)}"]
    return {
        "status": status,
        "valid": status != "invalid",
        "n_imag": len(negative),
        "negative_frequencies_cm1": negative,
        "flags": flags,
        "issues": issues,
        "weak_imag_threshold_cm1": WEAK_MINIMUM_IMAG_THRESHOLD_CM1,
    }
