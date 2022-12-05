"""A single image dataset for a specific satellite and specific month."""

import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import rasterio
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

# Our rasters contain no geolocation info, so silence this warning from rasterio
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# Setup S3 URLs and folder locations within the S3 bucket
S3_URL = "s3://drivendata-competition-biomassters-public-us"
S3_DATASOURCE = Path("/datasets/biomassters")
train_features_dir = S3_DATASOURCE / "train_features"
train_agbm_dir = S3_DATASOURCE / "train_agbm"
test_features_dir = S3_DATASOURCE / "test_features"

def load_raster(file_url: str) -> Tensor:
    """Returns the file URL and data as tensor."""
    # this is for local testing only ---
    S2_IMG_DIM = (11, 256, 256)
    if not os.path.exists(file_url):
        array = np.random.randn(S2_IMG_DIM)
    # --- end local test ---
    with rasterio.open(file_url) as f:
        array = f.read()
        if array.dtype == np.uint16:
            array = array.astype(np.int32)
    return torch.from_numpy(array)

class SingleImageDataset(Dataset):
    """This class implements a dataset that returns a single input and output image.
    Input image: Single satellite, single month.
    Output image: corresponding AGBM image.
    """

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

    def __init__(self, 
        metadata_file: str, 
        train =True, 
        transform=None, 
        target_transform=None, 
        satellite="S2", 
        month="april"):
        if metadata_file.endswith(".parquet"):
            metadata_df = pd.read_parquet(metadata_file)
        elif metadata_file.endswith(".csv"):
            metadata_df = pd.read_csv(metadata_file)
        else:
            print(f"Unsupported format for metadata file: {metadata_file}. "
                  "Only CSV and Parquet format files are supported.")

        self.satellite = satellite
        self.month = month
        self.transform = transform
        self.target_transform = target_transform
        self.month_id = self.month_map[month]
        self.train = train

        if train:
            self.chip_ids = self.metadata_df[self.metadata_df.split == "train"].chip_ids.unique()
        else:
            self.chip_ids = self.metadata_df[self.metadata_df.split == "test"].chip_ids.unique()

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.chip_ids)

    def __getitem__(self, idx):
        """Return a single (image, label) corresponding to idx."""
        # Input image
        if self.train:
            img_dir = train_features_dir
        else:
            img_dir = test_features_dir

        img_path = os.path.join(img_dir,
            self.chip_ids[idx],
            f"_{self.satellite}_{self.month_id}.tif")

        img_data = load_raster(img_path)[:10]  # only first 10 channels, leave out cloud coverage channel

        if self.transform is not None:
            img_data = self.transform(img_data)

        # Target image
        target_data = None
        if self.train:
            target_path = os.path.join(train_agbm_dir,
                self.chip_ids[idx],
                "_agbm.tif")
            target_data = load_raster(target_path)
            if self.target_transform is not None:
                target_data = self.target_transform(target_data)


        return {'image': img_data,
            'target': target_data,
            'chip_id': self.chip_ids[idx]}