"""Script for training TemporalSentinel2 model."""

import os
from pathlib import Path
from time import time

from biomasstry.datasets import TemporalSentinel2Dataset
from biomasstry.models import TemporalSentinel2Model
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
sentinel2_dataset = TemporalSentinel2Dataset()
torch.manual_seed(0)
train_size = int(0.8*len(sentinel2_dataset))
valid_size = len(sentinel2_dataset) - train_size
train_set, val_set = random_split(sentinel2_dataset, [train_size, valid_size])
print(f"Train samples: {len(train_set)} "
      f"Val. samples: {len(val_set)}")

# Model
model = TemporalSentinel2Model(n_samples=5,
    output_nc=1)

loss_module = nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

# DataLoaders
batch_size = 32
num_workers = 4
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

# Model Training
artifacts_dir = "/notebooks/artifacts"
model_name = "TemporalS2"
n_epochs = 1
date = "20230101"
save_path = artifacts_dir + f"/{date}_{model_name}_B{batch_size}_E{n_epochs}.pt"

# Kickoff training
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
