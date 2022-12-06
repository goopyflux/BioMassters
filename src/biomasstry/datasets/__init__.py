from .sentinel2 import Sentinel2

__all__ = (
    "Sentinel2",
)

# https://stackoverflow.com/questions/40018681
for module in __all__:
    globals()[module].__module__ = "biomasstry.datasets"