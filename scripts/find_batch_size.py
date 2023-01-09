"""Script that uses HuggingFace Accelerate library to find executable batch size.

References:
 - https://huggingface.co/docs/accelerate/usage_guides/memory
 - https://huggingface.co/docs/accelerate/package_reference/utilities#accelerate.find_executable_batch_size
 """

from accelerate import Accelerator, find_executable_batch_size
from biomasstry.datasets import TemporalSentinel1Dataset
from biomasstry.models import TemporalSentinelModel
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader

def training_function(batch_size):
    accelerator = Accelerator()

    @find_executable_batch_size(starting_batch_size=batch_size)
    def inner_training_loop(batch_size):
        nonlocal accelerator  # Ensure they can be used in our context
        accelerator.free_memory()  # Free all lingering references
        
        # Model
        model = TemporalSentinelModel(
            n_tsamples=12,
            input_nc=4,
            output_nc=1
        ).to(accelerator.device)

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

        # Sentinel-1 Dataset
        s1ds = TemporalSentinel1Dataset()

        # Split dataset into training and validation sets
        torch.manual_seed(0)
        train_size = int(0.8*len(s1ds))
        valid_size = len(s1ds) - train_size
        train_set, val_set = random_split(s1ds, [train_size, valid_size])
        print(f"Train samples: {len(train_set)} "
              f"Val. samples: {len(val_set)}")
        
        # DataLoaders
        batch_size = 32
        num_workers = 4
        train_dataloader = DataLoader(train_set,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=num_workers,
                                    pin_memory=True
                                    )

        eval_dataloader = DataLoader(val_set,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=num_workers,
                                    pin_memory=True
                                )

        # Prepare for training with Accelerate
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader
        )

        # Training Loop
        loss_function = nn.MSELoss(reduction='mean')  # Loss function
        nb_epochs = 1
        num_batches = len(eval_dataloader)
        train_metrics = []
        val_metrics = []
        for i in range(nb_epochs):
            train_metrics_epoch = []
            for batch in train_dataloader:
                optimizer.zero_grad()
                inputs = batch["image"]
                targets = batch["target"]
                outputs = model(inputs)
                loss = loss_function(outputs, targets)
                accelerator.backward(loss)
                optimizer.step()
                train_metrics_epoch.append(np.round(np.sqrt(loss.item()), 5))

            # Validation Loop
            val_loss = 0.0
            with torch.no_grad():
                for batch in eval_dataloader:
                    inputs = batch["image"]
                    targets = batch["target"]
                    predictions = model(inputs)
                    # Gather all predictions and targets
                    all_predictions, all_targets = accelerator.gather_for_metrics((predictions, targets))
                    val_loss += loss_function(predictions, targets).item()

            val_loss /= num_batches
            val_rmse = np.round(np.sqrt(val_loss), 5)
            print(f"Validation Error: \n RMSE: {val_rmse:>8f} \n")
            train_metrics.extend(train_metrics_epoch)
            val_metrics.append((len(train_metrics), val_rmse))
    inner_training_loop()

if __name__ == "__main__":
    training_function(batch_size=32)