"""A temporal image dataset for a specific satellite and months."""

import os
import pickle
from typing import Sequence, Optional, Callable, Dict, Any

from biomasstry.datasets.utils import load_raster, make_temporal_tensor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset


class TemporalSentinel1Dataset(Dataset):
    """Temporal Sentinel-1 Dataset (all months).
    The ESA Copernicus `Sentinel-1 <https://sentinel.esa.int/web/sentinel/missions/sentinel-1/overview>`_ mission comprises a constellation of
    polar-orbiting satellites, operating day and night performing `C-band
    Synthetic Aperture Radar (SAR) <https://www.earthdata.nasa.gov/learn/backgrounders/what-is-sar>`_ imaging, enabling them to acquire
    imagery regardless of the weather. The polar-orbits of SAR satellites
    mean that for half of their trajectory they are traveling from the
    north pole towards the south pole. This direction is referred to as a
    descending orbit. Conversely, when the satellite is traveling from the
    south pole towards the north pole, it is said to be in an ascending orbit.

    The provided S1 data include two bands from each of Sentinel-1's two
    satellites—"VV" and "VH"—for a total of four bands. These bands are
    captured from the sensor transmitting vertically polarized signal
    (represented by the first "V") and receiving either vertically (V)
    or horizontally (H) polarized signal.

    The values in these bands represent the energy that was reflected
    back to the satellite measured in decibels (dB). Pixel values can range
    from negative to positive values. A pixel value of -9999 indicates missing
    data. An advantage of Sentinel-1's use of SAR is that it can acquire data
    across day or night, under all weather conditions. Clouds or darkness do
    not impede the S1's ability to collect images.

    Finally, Sentinel-1 has a 6-day revisit orbit, which means that it returns
    to the same area about five times per month. We have provided a single
    composite image from S1 for each calendar month, which is generated by
    taking the mean across all images acquired by S1 for the patch during that
    time. For more details on how to interpret SAR data, participants might find
    it helpful to consult NASA's guide to SAR.
    """

    band_map = {
        "VVA": 1,
        "VHA": 2,
        "VVD": 3,
        "VHD": 4
    }

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

    temporal_months = ["april", "may", "june", "july", "august", "september"]

    # Setup S3 URLs and folder locations within the S3 bucket
    # S3_URL = "s3://drivendata-competition-biomassters-public-us"
    S3_URL = "/datasets/biomassters"
    
    def __init__(self, 
        metadata_file: str = "",
        data_url: str = "",
        months: Sequence[str] =[],
        bands: Sequence[str] = [],
        train: bool = True, 
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None, 
        target_transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    ) -> None:
        """ Initialize a new instance of the Sentinel-2 Dataset.
        Args:
        """
        if metadata_file:
            self.metadata_file = metadata_file
        else:
            self.metadata_file = "/notebooks/data/metadata_parquet/metadata_chipid_split.parquet"
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

        self.months = months if months else self.temporal_months  # list(self.month_map.keys())
        if bands:
            self.band_indexes = np.asarray([self.band_map[band] for band in bands])
        else:
            self.band_indexes = np.arange(1, 5)  # All bands
        self.transform = transform
        self.target_transform = target_transform
        self._lst, self._addr = self._get_chip_ids()
        
    def _get_chip_ids(self):
        def _serialize(data):
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        metadata_df = pd.read_parquet(self.metadata_file)
        if self.train:
            chip_ids = metadata_df[metadata_df.split == "train"].chip_id.unique().astype(np.str_)
        else:
            chip_ids = metadata_df[metadata_df.split == "test"].chip_id.unique().astype(np.str_)

        lst = [_serialize(x) for x in chip_ids]
        addr = np.asarray([len(x) for x in lst], dtype=np.int64)
        addr = np.cumsum(addr)
        lst = np.concatenate(lst)
        return torch.from_numpy(lst), torch.from_numpy(addr)

    def _get_image_paths(self, chip_id):
        return np.asarray([self.feaures_dir + f"/{chip_id}_S1_{self.month_map[m]}.tif"
                          for m in self.months])

    def __len__(self):
        """Return the length of the dataset."""
        return len(self._addr)

    def __getitem__(self, idx):
        """Return a single (image, label) corresponding to idx."""
        start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
        end_addr = self._addr[idx].item()
        bytes = memoryview(self._lst[start_addr:end_addr].numpy())
        chip_id = pickle.loads(bytes)

        # Input image
        img_paths = self._get_image_paths(chip_id)
        
        # Create temporal tensor of size TxCxWxH
        timg_data = make_temporal_tensor(img_paths, self.band_indexes)

        # TODO Update transform to work with temporal tensor.
        if self.transform is not None:
            timg_data = [self.transform(img) for img in timg_data]

        # Target image
        target_data = None
        if self.train:
            target_path = self.targets_dir + f"/{chip_id}_agbm.tif"
            target_data = load_raster(target_path)
            if self.target_transform is not None:
                target_data = self.target_transform(target_data)

        # return {'image': timg_data,
            # 'target': target_data,
            # 'chip_id': self.chip_ids[idx]}
        return timg_data, target_data, chip_id

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