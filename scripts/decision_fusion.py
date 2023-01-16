"""Script to perform decision fusion of multi-model predictions."""

import os

import numpy as np
from PIL import Image
import rasterio
from tqdm import tqdm

PREDICTIONS_DIR = "artifacts/decision_fusion_mean"

def save_agbm(agbm_pred, image_path):
    im = Image.fromarray(agbm_pred)
    # save_path = os.path.join(PREDICTIONS_DIR, f'{chipid}_agbm.tif')
    im.save(image_path, format='TIFF', save_all=True)

# Predictions Folders
prediction_folders = ["artifacts/s2_all_submission",
    "artifacts/s1_asc_submission",
    "artifacts/s1_dsc_submission"
    ]

# s1_asc_dir = "artifacts/s1_asc_submission"
# s1_dsc_dir = "artifacts/s1_dsc_submission"
# s2_all_dir = "artifacts/s2_all_submission"

# list files
agbm_files = os.listdir(prediction_folders[0])

# average predictions
for i, agbm_file in tqdm(enumerate(agbm_files)):
    image_data = []
    for folder in prediction_folders[:-1]:
        image_file = os.path.join(folder, agbm_file)
        image_data.append(rasterio.open(image_file).read())

    image_array = np.asarray(image_data).squeeze()
    mean_prediction = image_array.mean(axis=0)
    prediction_file = os.path.join(PREDICTIONS_DIR, agbm_file)
    save_agbm(mean_prediction, prediction_file)
