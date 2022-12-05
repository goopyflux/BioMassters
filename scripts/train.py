
from pathlib import Path

from biomasstry.data import SingleImageDataset
from biomasstry.models import Sentinel2Model
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader
import wandb



METADATA_FILE = Path(".").resolve().parent / "data/metadata_parquet/features_metadata_slim.parquet"
MODEL_PATH = Path(".").resolve().parent / "artifacts/resnet50-sentinel2.pt"

train_dataset = SingleImageDataset(METADATA_FILE)
train_size = int(0.8*len(train_dataset))
valid_size = len(train_dataset) - train_size
train_set, val_set = torch.utils.data.random_split(train_dataset, [train_size, valid_size])
train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers = 6)
valid_dataloader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers = 6)

test_dataset = SingleImageDataset(METADATA_FILE, train=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers = 6)

base_model = smp.Unet(
    encoder_name="resnet50",       
    in_channels=10,                 
    classes=1,                     
)

base_model.encoder.load_state_dict(torch.load())
s2_model = Sentinel2Model(base_model)
wandb_logger = WandbLogger(name='Sentinel_2_ResNet50', project='BioMassters_baseline')

trainer = Trainer(
    accelerator="gpu",
    max_epochs=20,
    logger=[wandb_logger],
)

# Train the model ⚡
trainer.fit(s2_model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)