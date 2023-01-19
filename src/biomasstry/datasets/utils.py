"""Dataset Utility Functions."""

import warnings

import fsspec
import numpy as np
import rasterio
from rasterio.io import MemoryFile
import torch
from torch import Tensor


# Our rasters contain no geolocation info, so silence this warning from rasterio
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

def load_raster(file_url: str, indexes=None) -> Tensor:
    """Returns the TIF image as tensor.
    file_url: specifies full path for the TIF file.
    indexes: [List[int]] specific indexes to be read. 
             First index always starts at 1 (not 0).
             None is the same as all indexes (bands)
    """
    storage_options = {'anon': True}
    with fsspec.open(file_url, **storage_options).open() as f:
        raw_bytes = f.read()
        # Save bytes to array
        with MemoryFile(raw_bytes) as memfile:
            with memfile.open() as buffer:
                if indexes is None:
                    array = buffer.read()
                else:
                    array = buffer.read(indexes)
                if array.dtype == np.uint16:
                    array = array.astype(np.float)
    return torch.tensor(array, dtype=torch.float32)

def make_temporal_tensor(image_paths, band_indexes):
    timg_data = [load_raster(img_path, indexes=band_indexes) for img_path in image_paths]

    # Stack temporally to create a TxCxWxH dataset
    timg_data = torch.stack(timg_data, dim=0)
    
    return timg_data
