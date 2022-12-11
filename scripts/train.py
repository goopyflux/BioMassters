"""Script for training the model."""

import os
from pathlib import Path
from time import time

from biomasstry.datasets import Sentinel2
from biomasstry.models import FCN
from biomasstry.models.utils import run_training
import numpy as np
import pandas as pd
import seaborn as sns
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split


sns.set()

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"Device: {device}")

# Train Dataset
sentinel2_dataset = Sentinel2()
torch.manual_seed(0)
train_size = int(0.8*len(sentinel2_dataset))
valid_size = len(sentinel2_dataset) - train_size
train_set, val_set = random_split(sentinel2_dataset, [train_size, valid_size])
print(f"Train samples: {len(train_set)} "
      f"Val. samples: {len(val_set)}")

# Model
in_channels = train_set[0]['image'].shape[0]
model = FCN(in_channels=in_channels,
    classes=1).to(device)

loss_module = nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

# Model Training
artifacts_dir = "/notebooks/artifacts"
batch_size = 64
num_workers = 4
# DataLoaders
train_dataloader = DataLoader(train_set,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            pin_memory=True
                            )

val_dataloader = DataLoader(val_set,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True
                        )

save_path = artifacts_dir + "/FCN_10bandS2Apr_batch_AGBMLinear_10epoch_10DEC.pt"

# Kickoff training
n_epochs = 10
# start = time()
metrics = run_training(model=model,
                    loss_module=loss_module,
                    optimizer=optimizer,
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    save_path=save_path,
                    n_epochs=n_epochs)

# Save the metrics to a file
train_metrics_df = pd.DataFrame(metrics['training'], columns=["step", "score"])
val_metrics_df = pd.DataFrame(metrics["validation"], columns=["step", "score"])
train_metrics_df.to_csv(artifacts_dir + "/train_metrics.csv")
val_metrics_df.to_csv(artifacts_dir + "/val_metrics.csv")

# Test Dataset
# test_dataset = Sentinel2(train=False)
# test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers = 6)