#!/usr/bin/env python
# coding: utf-8

# # Use HuggingFace Accelerate to Train Model on Temporal (monthly) Sentinel Data

# ## Imports

# In[1]:


from time import time


# In[2]:


from accelerate import Accelerator, notebook_launcher, init_empty_weights
from accelerate.utils import set_seed
from biomasstry.datasets import TemporalSentinel2Dataset, TemporalSentinel1Dataset
from biomasstry.models import TemporalSentinelModel, UTAE
from biomasstry.models.unet_tae import ConvBlock
# from biomasstry.models.utils import run_training
import numpy as np
import pandas as pd
from pynvml import *
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from transformers import TrainingArguments, Trainer, logging
from tqdm.notebook import tqdm


# In[3]:


logging.set_verbosity_error()


# ## Utility Functions

# In[4]:


# Utility functions for printing GPU utilization
def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


# In[5]:


print_gpu_utilization()


# ## Dataset and DataLoaders

# In[6]:


def get_dataloaders(dataset: str, batch_size: int=16, num_workers: int=0):
    """Return train and eval DataLoaders with specified batch size.
    
    dataset: str
        Dataset identifier. Must be one of "Sentinel-1A", "Sentinel-1D" or "Sentinel-2All"
    batch_size: int
        batch size for each batch.
    """
    # If True, access directly S3.
    # If False, assume data is mounted and available under '/datasets/biomassters'
    S3_DIRECT = False
    if S3_DIRECT:
        data_url="s3://drivendata-competition-biomassters-public-us"
    else:
        data_url = ""

    if dataset == "Sentinel-1A": # Sentinel-1 Ascending only
        ds = TemporalSentinel1Dataset(data_url=data_url, bands=["VVA", "VHA"])
    elif dataset == "Sentinel-1D": # Sentinel-1 Descending only
        ds = TemporalSentinel1Dataset(data_url=data_url, bands=["VVD", "VHD"])
    elif dataset == "Sentinel-2all":
        ds = TemporalSentinel2Dataset(data_url=data_url)
    else:
        print("Unrecognized dataset identifier. Must be one of 'Sentinel-1A', 'Sentinel-1D' or 'Sentinel-2all'")
        return None, None

    train_size = int(0.8*len(ds))
    valid_size = len(ds) - train_size
    train_set, val_set = random_split(ds, [train_size, valid_size])
    print(f"Train samples: {len(train_set)} "
        f"Val. samples: {len(val_set)}")

    # DataLoaders
    train_dataloader = DataLoader(train_set,
                        batch_size=batch_size,
                        shuffle=True,
                        pin_memory=True,
                        num_workers=num_workers)
    eval_dataloader = DataLoader(val_set,
                        batch_size=batch_size * 2,
                        shuffle=False,
                        pin_memory=True,
                        num_workers=num_workers)
    return train_dataloader, eval_dataloader


# ## Training Loop

# In[7]:


def training_loop(dataset: str,
                  mixed_precision: str="fp16",
                  seed: int=123,
                  batch_size: int=16,
                  gradient_accumulation_steps: int=4,
                  nb_epochs=2,
                  train_mode: str=""
    ):
    """Main Training and Evaluation Loop to be called by accelerator.notebook_launcher()."""
    print(f"Args: {mixed_precision}, {seed}, {batch_size}, "
          f"{gradient_accumulation_steps}, {nb_epochs}, {train_mode}")
    
    # Set random seed
    set_seed(seed)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")
    
    # Initialize Accelerator
    # accelerator = Accelerator(mixed_precision=mixed_precision,
        # gradient_accumulation_steps=gradient_accumulation_steps)

    # Build DataLoaders
    train_dataloader, eval_dataloader = get_dataloaders(dataset, batch_size=batch_size)

    # Assign model inputs based on dataset
    if dataset == "Sentinel-1A":
        input_nc = 2
        n_tsamples = 6
    elif dataset == "Sentinel-1D":
        input_nc = 2
        n_tsamples = 6
    elif dataset == "Sentinel-2all":
        input_nc = 10
        n_tsamples = 5
    else:
        return

    # Create model
    if train_mode == "tune":
        # with init_empty_weights():
            # model = UTAE(10, out_conv=[32, 20])  # Initialize the original model & load pre-trained weights
        model = UTAE(10, out_conv=[32, 20])  # Initialize the original model & load pre-trained weights
        saved_dict = torch.load(pretrained_weights_path)
        model.load_state_dict(saved_dict["state_dict"])
        model.out_conv = ConvBlock([32, 32, 1], padding_mode="reflect")  # Modify the last layer
    else:
        model = UTAE(input_nc)  # modify output layer to predict AGBM
        if train_mode == "resume":
            state_dict = torch.load(saved_state_path, map_location=accelerator.device)
            model.load_state_dict(state_dict)
    
    model = model.to(device)
    loss_function = nn.MSELoss(reduction='mean').to(device)  # Loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)  # Optimizer
    
    # Prepare everything to use accelerator
    # Maintain order while unpacking
    # model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model,
    #                                                            optimizer,
    #                                                            train_dataloader,
    #                                                            eval_dataloader)

    # Training loop
    min_valid_metric = np.inf
    for i in tqdm(range(nb_epochs)):
        # accelerator.print(f"Epoch {i+1}")
        print(f"Epoch {i+1}")
        epoch_start = time()
        # accelerator.print(f"Training")
        print(f"Training")
        for batch in tqdm(train_dataloader):
            # with accelerator.accumulate(model):
            optimizer.zero_grad()
            inputs = batch["image"].to(device)
            targets = batch["target"].to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            # accelerator.backward(loss)
            loss.backward()
            optimizer.step()

        epoch_end = time()
        # accelerator.print(f"  Training time: {epoch_end - epoch_start}")
        print(f"  Training time: {epoch_end - epoch_start}")
        
        # Save Model State Dict after each epoch in order to continue training later
        # unwrap_model = accelerator.unwrap_model(model)  # Unwrap the Accelerator model
        train_model_path = save_path[:-3] + f"_E{i+1}.pt"
        # accelerator.save(unwrap_model.state_dict(), train_model_path)
        # accelerator.print(f"  Model file path: {train_model_path}")
        torch.save(model.state_dict(), train_model_path)
        print(f"  Model file path: {train_model_path}")
     
        # Validation Loop
        val_loss = 0.0
        num_elements = 0
        # accelerator.print(f"Validation")
        print(f"Validation")
        for batch in tqdm(eval_dataloader):
            inputs = batch["image"].to(device)
            targets = batch["target"].to(device)
            with torch.no_grad():
                predictions = model(inputs)
            # Gather all predictions and targets
            # all_predictions, all_targets = accelerator.gather_for_metrics((predictions, targets))
            # num_elements += all_predictions.shape[0]
            # val_loss += loss_function(all_predictions, all_targets).item()
            val_loss += loss_function(predictions, targets).item()

        val_loss /= len(eval_dataloader)
        val_rmse = np.round(np.sqrt(val_loss), 5)
        # accelerator.print(f"  Validation RMSE: {val_rmse:>8f}")
        print(f"  Validation RMSE: {val_rmse:>8f}")
        # check validation score, if improved then save model
        if min_valid_metric > val_rmse:
            # accelerator.print(f"  Validation RMSE Decreased({min_valid_metric:.6f}--->{val_rmse:.6f})")
            print(f"  Validation RMSE Decreased({min_valid_metric:.6f}--->{val_rmse:.6f})")
            min_valid_metric = val_rmse

            # Saving Model State Dict
            # unwrap_model = accelerator.unwrap_model(model)  # Unwrap the Accelerator model
            # accelerator.save(unwrap_model.state_dict(), best_model_path)
            # accelerator.print(f"  Best Model file path: {best_model_path}")
            torch.save(model.state_dict(), best_model_path)
            print(f"  Best Model file path: {best_model_path}")


# In[8]:


dataset = "Sentinel-2all"
mixed_precision = "fp16"
seed = 123
batch_size = 8
gradient_accumulation_steps = 4
nb_epochs = 10
train_mode = "tune"

artifacts_dir = "/notebooks/artifacts"
model_name = "UTAE"
date = "20230117"
pretrained_weights_path = artifacts_dir + "/pretrained_utae/f1model.pth.tar"  # for fine tuning
saved_state_path = artifacts_dir + "/20230112_UTAE_S2_B32_E20.pt"  # for resuming training

save_path = artifacts_dir + (f"/{date}_{model_name}_{dataset}_B"
        f"{batch_size * gradient_accumulation_steps}.pt")
best_model_path = save_path[:-3] + "_BEST.pt"

train_args = (dataset, mixed_precision, seed, batch_size, gradient_accumulation_steps, nb_epochs, train_mode)
# notebook_launcher(training_loop, train_args, num_processes=1)
training_loop(*train_args)


# In[ ]:


print(train_args)

##### Save the metrics to a file
train_metrics_zipped = list(zip(np.arange(0, len(train_metrics)), train_metrics))
metrics = {'training': train_metrics_zipped, 'validation': val_metrics}
train_metrics_df = pd.DataFrame(metrics['training'], columns=["step", "score"])
val_metrics_df = pd.DataFrame(metrics["validation"], columns=["step", "score"])
train_metrics_df.to_csv(artifacts_dir + "/train_metrics.csv")
val_metrics_df.to_csv(artifacts_dir + "/val_metrics.csv")