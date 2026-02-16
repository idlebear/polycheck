"""
Warp backend exports for polycheck.

Usage:
    import polycheck.warp as pw
"""

from .poly_warp import (
    contains,
    visibility,
    visibility_from_region,
    visibility_from_real_region,
    faux_scan,
)


def sensor_visibility_from_region(*args, **kwargs):
    raise NotImplementedError(
        "sensor_visibility_from_region is not implemented in the Warp backend yet."
    )


def sensor_visibility_from_real_region(*args, **kwargs):
    raise NotImplementedError(
        "sensor_visibility_from_real_region is not implemented in the Warp backend yet."
    )


name = "polycheck.warp"
__all__ = [
    "contains",
    "visibility",
    "visibility_from_region",
    "visibility_from_real_region",
    "sensor_visibility_from_region",
    "sensor_visibility_from_real_region",
    "faux_scan",
]
