"""Available models in biomasstry package."""

from .fcn import FCN
from .temporal_model import TemporalSentinelModel
from .unet_tae import UTAE

__all__ = (
    "FCN",
    "TemporalSentinelModel",
    "UTAE",
)

# https://stackoverflow.com/questions/40018681
for module in __all__:
    globals()[module].__module__ = "biomasstry.models"