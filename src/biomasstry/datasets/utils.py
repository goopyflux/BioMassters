"""Dataset Utility Functions."""

import warnings

import fsspec
import numpy as np
import rasterio
from rasterio.io import MemoryFile
import torch


# Our rasters contain no geolocation info, so silence this warning from rasterio
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

def load_raster(file_url: str, indexes=None) -> torch.Tensor:
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
    # Stack temporally to create a TxCxWxH dataset
    im0 = load_raster(image_paths[0], indexes=band_indexes.tolist())
    im1 = load_raster(image_paths[1], indexes=band_indexes.tolist())
    im2 = load_raster(image_paths[2], indexes=band_indexes.tolist())
    im3 = load_raster(image_paths[3], indexes=band_indexes.tolist())
    im4 = load_raster(image_paths[4], indexes=band_indexes.tolist())
    
    # return torch.stack([load_raster(img_path, indexes=band_indexes.tolist()) 
    #                          for img_path in image_paths], dim=0)
    return torch.stack((im0, im1, im2, im3, im4), dim=0)
