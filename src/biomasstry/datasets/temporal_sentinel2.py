"""A single image dataset for a specific satellite and specific month."""

import os
from pathlib import Path
from typing import Sequence, Optional, Callable, Dict, Any
import warnings

import fsspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.io import MemoryFile
import fsspec
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

# Our rasters contain no geolocation info, so silence this warning from rasterio
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


def load_raster(file_url: str) -> Tensor:
    """Returns the data as tensor."""
    # this is for local testing only ---
    # S2_IMG_DIM = (11, 256, 256)
    # if not os.path.exists(file_url):
        # array = np.random.randn(S2_IMG_DIM)
    # --- end local test ---
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


class TemporalSentinel2Dataset(Dataset):
    """Temporal Sentinel-2 Dataset (only months April - August).
    
    Sentinel-2 (S2) is a high-resolution imaging mission that monitors vegetation,
    soil, water cover, inland waterways, and coastal areas. S2 satellites have a
    Multispectral Instrument (MSI) on board that collects data in the visible,
    near-infrared, and short-wave infrared portions of the electromagnetic spectrum.
    Sentinel-2 measures spectral bands that range from 400 to 2400 nanometers.
    Sentinel-2 has a 6-day revisit orbit, which means that it returns to the same 
    area about five times per month. The best image for each month is selected from 
    the S2 data.

    The following 11 bands are provided for each S2 image:
    B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, and CLP (a cloud probability layer).
    See the `Sentinel-2 guide <https://docs.sentinel-hub.com/api/latest/data/sentinel-2-l2a/#available-bands-and-data>`_ for a description of each band.

    The CLP band — cloud probability — is provided because S2 cannot penetrate clouds.
    The cloud probability layer has values from 0-100, indicating the percentage
    probability of cloud cover for that pixel. In some images, this layer may have
    a value of 255, which indicates that the layer has been obscured due to excessive
    noise.

    For information on the satellite data and its sources, check out the competiton `About Page <https://www.drivendata.org/competitions/99/biomass-estimation/page/535/>`_.
    """

    all_bands = [
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        "B11",
        "B12",
        "CLP"
    ]

    month_map = {
        "september": "00",
        "october": "01",
        "november": "02",
        "december": "03",
        "january": "04",
        "february": "05",
        "march": "06",
        "april": "07",
        "may": "08",
        "june": "09",
        "july": "10",
        "august": "11"
    }

    temporal_months = ["april", "may", "june", "july", "august"]

    # Setup S3 URLs and folder locations within the S3 bucket
    # S3_URL = "s3://drivendata-competition-biomassters-public-us"
    S3_URL = "/datasets/biomassters"
    metadata_file = "/notebooks/data/metadata_parquet/features_metadata_slim.parquet"
    
    def __init__(self, 
        metadata_file: str = "",
        data_url: str = "",
        bands: Sequence[str] = [], 
        months: Sequence[str] =[],
        train: bool = True, 
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None, 
        target_transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    ) -> None:
        """ Initialize a new instance of the Sentinel-2 Dataset.
        Args:
        """
        if not metadata_file:
            metadata_file = self.metadata_file
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"File {metadata_file} not found! "
                                    "Please check the path and make sure the file exists."
                                   )

        self.data_url = data_url
        self.train = train
        # Data URL resolution
        if not self.data_url:
            self.data_url = self.S3_URL
        if self.train:
            self.feaures_dir = self.data_url + "/train_features"
            self.targets_dir = self.data_url + "/train_agbm"
        else:
            self.feaures_dir = self.data_url + "/test_features"
            self.targets_dir = ""

        self.bands = bands if bands else self.all_bands

        if metadata_file.endswith(".parquet"):
            metadata_df = pd.read_parquet(metadata_file)
        elif metadata_file.endswith(".csv"):
            metadata_df = pd.read_csv(metadata_file)
        else:
            raise Exception(f"Unsupported format for metadata file: {metadata_file}. "
                  "Only CSV and Parquet format files are supported.")

        self.months = months if months else self.temporal_months
        self.transform = transform
        self.target_transform = target_transform
        
        if train:
            self.chip_ids = metadata_df[metadata_df.split == "train"].chip_id.unique()
        else:
            self.chip_ids = metadata_df[metadata_df.split == "test"].chip_id.unique()

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.chip_ids)

    def __getitem__(self, idx):
        """Return a single (image, label) corresponding to idx."""
        # Input image
        img_paths = [self.feaures_dir + f"/{self.chip_ids[idx]}_S2_{self.month_map[m]}.tif" 
            for m in self.months]
        timg_data = [load_raster(img_path)[:10] for img_path in img_paths]  # only first 10 channels, leave out cloud coverage channel

        if self.transform is not None:
            timg_data = [self.transform(img) for img in timg_data]

        # Target image
        target_data = None
        if self.train:
            target_path = self.targets_dir + f"/{self.chip_ids[idx]}_agbm.tif"
            target_data = load_raster(target_path)
            if self.target_transform is not None:
                target_data = self.target_transform(target_data)

        return {'image': timg_data,
            'target': target_data,
            'chip_id': self.chip_ids[idx]}

    def plot(
        self,
        sample: Dict[str, Any],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.
        Args:
            sample: a sample returned by :meth:`RasterDataset.__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle
        Returns:
            a matplotlib Figure with the rendered sample
        Raises:
            ValueError: if the RGB bands are not included in ``self.bands``
        .. versionchanged:: 0.3
           Method now takes a sample dict, not a Tensor. Additionally, possible to
           show subplot titles and/or use a custom suptitle.
        """
        rgb_indices = []
        for band in self.RGB_BANDS:
            if band in self.bands:
                rgb_indices.append(self.bands.index(band))
            else:
                raise ValueError("Dataset doesn't contain some of the RGB bands")

        image = sample["image"][rgb_indices].permute(1, 2, 0)
        image = torch.clamp(image / 2000, min=0, max=1)

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))

        ax.imshow(image)
        ax.axis("off")

        if show_titles:
            ax.set_title("Image")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return 