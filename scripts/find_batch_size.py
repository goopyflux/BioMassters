"""Script that uses HuggingFace Accelerate library to find executable batch size.

References:
 - https://huggingface.co/docs/accelerate/usage_guides/memory
 - https://huggingface.co/docs/accelerate/package_reference/utilities#accelerate.find_executable_batch_size
 """

from time import time

from accelerate import Accelerator, find_executable_batch_size
from biomasstry.datasets import TemporalSentinel1Dataset, TemporalSentinel2Dataset
from biomasstry.models import TemporalSentinelModel, UTAE
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader

USE_SENTINEL_1 = False

def training_function(batch_size):
    accelerator = Accelerator(gradient_accumulation_steps=4, mixed_precision='fp16')

    @find_executable_batch_size(starting_batch_size=batch_size)
    def inner_training_loop(batch_size):
        nonlocal accelerator  # Ensure they can be used in our context
        accelerator.free_memory()  # Free all lingering references
        
        # Sentinel Dataset
        if USE_SENTINEL_1:
            sds = TemporalSentinel1Dataset(bands=["VVA", "VHA"])
            input_nc = 2
            n_tsamples = 6
        else:
            sds = TemporalSentinel2Dataset()
            input_nc = 10
            n_tsamples = 5
        
        # Model
        # model = TemporalSentinelModel(
        #     n_tsamples=n_tsamples,
        #     input_nc=input_nc,
        #     output_nc=1
        # ).to(accelerator.device)
        model = UTAE(input_nc).to(accelerator.device)

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


        # Split dataset into training and validation sets
        torch.manual_seed(0)
        train_size = int(0.8*len(sds))
        valid_size = len(sds) - train_size
        train_set, val_set = random_split(sds, [train_size, valid_size])
        print(f"Train samples: {len(train_set)} "
              f"Val. samples: {len(val_set)}")
        
        # DataLoaders
        num_workers = 6
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
        print(f"Starting training with batch size = {batch_size}")
        loss_function = nn.MSELoss(reduction='mean')  # Loss function
        num_batches = len(eval_dataloader)
        train_metrics = []
        val_metrics = []
        start_epoch = time()
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
        
        end_epoch = time()
        print(f"Time for one epoch on batch size {batch_size}: {end_epoch - start_epoch}")
        
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
    start_all = time()
    training_function(batch_size=16)
    end_all = time()
    print(f"Total time: {end_all - start_all}")