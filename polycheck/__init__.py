import importlib

from .polycheck import (
    contains,
    visibility,
    visibility_from_region,
    visibility_from_real_region,
    sensor_visibility_from_region,
    sensor_visibility_from_real_region,
    faux_scan,
)

name = "polycheck"
__all__ = [
    "contains",
    "visibility",
    "visibility_from_region",
    "visibility_from_real_region",
    "sensor_visibility_from_region",
    "sensor_visibility_from_real_region",
    "faux_scan",
]


def __getattr__(name):
    if name == "warp":
        # Lazy-load Warp backend so importing `polycheck` does not require warp-lang.
        return importlib.import_module(".warp", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
