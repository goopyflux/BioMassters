"""Script that uses pre-trained models to save predicted AGBM output images.

The pre-trained models are generated from training on Sentinel-2 images filtered
by month. Only months April - August are considered for training (5 models total)."""

from pathlib import Path

from biomasstry.datasets import Sentinel2
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

ROOT_DIR = Path("/notebooks")
MODELS_DIR = ROOT_DIR / "artifacts"
PREDICTIONS_DIR = ROOT_DIR / "data/S2_Monthly_AGBM"
# S3_URL = "s3://drivendata-competition-biomassters-public-us"

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

device = "cpu" if not torch.cuda.is_available() else "cuda"

def load_model(month, in_channels=10):
    """Return the pre-trained model for the specified month."""
    model_path = MODELS_DIR / f"UNET_resnet50_10bandS2{month}_batch64_AGBMLinear_20epoch_10DEC.pt"
    model = smp.Unet(
    encoder_name="resnet50",
    encoder_weights=None, # 'imagenet' weights don't seem to help so start clean 
    in_channels=in_channels,                 
    classes=1,                     
    ).to(device) 
    model.load_state_dict(torch.load(model_path))
    return model

def get_dataloader(month):
    """Returns a DataLoader for Sentinel-2 images from specified month."""
    # Dataset and DataLoader
    sen2dataset = Sentinel2(month=month)
    batch_size = 32
    num_workers = 6
    sen2dl = DataLoader(sen2dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True)

    return sen2dl

def save_prediction(month, dl, model):
    """Saves predictions to image."""
    nb_batches = len(dl)
    month_id = month_map[month]
    with torch.no_grad():
        for batch in tqdm(dl, total=nb_batches):
            X = batch["image"].to(device)
            ids = batch["chip_id"]

            # predicted outputs
            pred = model(X)
            for i in range(pred.size(0)):
                file_name = PREDICTIONS_DIR / f"{ids[i]}_S2_{month_id}_agbm.tif"
                save_image(pred[i, 0, :, :], file_name)


def main():
    months = {"april", "may", "june", "july"}  # , "august"}
    print("Saving monthly predictions...")
    for month in tqdm(months):
        print(f"Month: {month}")
        # Get DataLoader
        dl = get_dataloader(month)

        # Load model
        model = load_model(month)
        model.eval()

        # Save predictions
        save_prediction(month, dl, model)
        break

if __name__ == "__main__":
    main()