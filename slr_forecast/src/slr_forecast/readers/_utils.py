"""
Shared utility functions for data readers.

Sign convention: positive = sea level rise (SLR) throughout.
Internal units: meters (sea level), degC (temperature), decimal years (time).
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd


def decimal_year_to_datetime(decimal_year: float) -> datetime:
    """Convert decimal year to datetime object."""
    year = int(decimal_year)
    remainder = decimal_year - year
    base = datetime(year, 1, 1)
    next_year = datetime(year + 1, 1, 1)
    return base + (next_year - base) * remainder


def datetime_to_decimal_year(dt: datetime) -> float:
    """Convert datetime to decimal year."""
    year = dt.year
    base = datetime(year, 1, 1)
    next_year = datetime(year + 1, 1, 1)
    return year + (dt - base).total_seconds() / (next_year - base).total_seconds()
