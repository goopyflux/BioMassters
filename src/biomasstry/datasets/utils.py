"""Dataset Utility Functions."""

import warnings

import fsspec
# import numpy as np
import rasterio
from rasterio.io import MemoryFile
import torch
from torch import Tensor


# Our rasters contain no geolocation info, so silence this warning from rasterio
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

def load_raster(file_url: str) -> Tensor:
    """Returns the TIF image as tensor."""
    storage_options = {'anon': True}
    with fsspec.open(file_url, **storage_options).open() as f:
        raw_bytes = f.read()
        # Save bytes to array
        with MemoryFile(raw_bytes) as memfile:
            with memfile.open() as buffer:
                array = buffer.read()
                # if array.dtype == np.uint16:
                    # array = array.astype(np.float)
    return torch.tensor(array, dtype=torch.float32)
