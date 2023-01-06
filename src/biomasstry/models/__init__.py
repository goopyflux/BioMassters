"""Available models in biomasstry package."""

from .fcn import FCN
from .temporal_model import TemporalSentinelModel

__all__ = (
    "FCN",
    "TemporalSentinelModel",
)

# https://stackoverflow.com/questions/40018681
for module in __all__:
    globals()[module].__module__ = "biomasstry.models"