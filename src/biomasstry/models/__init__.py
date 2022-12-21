"""Available models in biomasstry package."""

from .fcn import FCN
from .temporal_model import TemporalSentinel2Model

__all__ = (
    "FCN",
    "TemporalSentinel2Model",
)

# https://stackoverflow.com/questions/40018681
for module in __all__:
    globals()[module].__module__ = "biomasstry.models"