"""Script to analyze output images.

Compute mean, min, max, median, quartiles, unique values, etc.
"""

import os
import warnings

import fsspec
import numpy as np
import pandas as pd
import rasterio.io import MemoryFile

S3_URL = "s3://drivendata-competition-biomassters-public-us/"
agbm_path = S3_URL + "train_agbm"

# Our rasters contain no geolocation info, so silence this warning from rasterio
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

def load_raster(file_url: str) -> Tensor:
    """Returns the data as tensor."""
    storage_options = {'anon': True}
    with fsspec.open(file_url, **storage_options).open() as f:
        raw_bytes = f.read()
        # Save bytes to array
        with MemoryFile(raw_bytes) as memfile:
            with memfile.open() as buffer:
                array = buffer.read()
                if array.dtype == np.uint16:
                    array = array.astype(np.float)
    return torch.tensor(array, dtype=torch.float32)
