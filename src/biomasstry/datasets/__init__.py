from .sentinel2 import Sentinel2
from .temporal_sentinel2 import TemporalSentinel2Dataset

__all__ = (
    "Sentinel2",
    "TemporalSentinel2Dataset",
)

# https://stackoverflow.com/questions/40018681
for module in __all__:
    globals()[module].__module__ = "biomasstry.datasets"