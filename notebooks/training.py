from time import time

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
from tqdm import tqdm

@profile
def get_dataloaders(chip_ids, dataset: str, batch_size: int=8, num_workers: int=4):
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

    random_perm = np.random.permutation(len(chip_ids))
    cut = int(0.8 * len(chip_ids))
    train_split = random_perm[:cut]
    eval_split = random_perm[cut:]

    train_ds = eval_ds = None
    if dataset == "Sentinel-1A": # Sentinel-1 Ascending only
        ds = TemporalSentinel1Dataset(data_url=data_url, bands=["VVA", "VHA"])
    elif dataset == "Sentinel-1D": # Sentinel-1 Descending only
        ds = TemporalSentinel1Dataset(data_url=data_url, bands=["VVD", "VHD"])
    elif dataset == "Sentinel-2all":
        train_ds = TemporalSentinel2Dataset([chip_ids[i] for i in train_split], data_url=data_url)
        eval_ds = TemporalSentinel2Dataset([chip_ids[i] for i in eval_split], data_url=data_url)
    else:
        print("Unrecognized dataset identifier. Must be one of 'Sentinel-1A', 'Sentinel-1D' or 'Sentinel-2all'")
        return None, None

    print(f"Train samples: {len(train_ds)} "
        f"Val. samples: {len(eval_ds)}")

    # DataLoaders
    pin_memory = False
    train_dataloader = DataLoader(train_ds,
                        batch_size=batch_size,
                        shuffle=True,
                        pin_memory=pin_memory,
                        num_workers=num_workers)
    eval_dataloader = DataLoader(eval_ds,
                        batch_size=batch_size,
                        shuffle=False,
                        pin_memory=pin_memory,
                        num_workers=num_workers)
    return train_dataloader, eval_dataloader

@profile
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
    

    # Metadata
    metadata_file = "/notebooks/data/metadata_parquet/features_metadata_slim.parquet"
    metadata_df = pd.read_parquet(metadata_file)
    chip_ids = metadata_df[metadata_df.split == "train"].chip_id.unique().tolist()

    artifacts_dir = "/notebooks/artifacts"
    model_name = "UTAE"
    date = "20230118"
    pretrained_weights_path = artifacts_dir + "/pretrained_utae/f1model.pth.tar"  # for fine tuning

    saved_state_path = artifacts_dir + "/20230112_UTAE_S2_B32_E20.pt"  # for resuming training
    save_path = artifacts_dir + (f"/{date}_{model_name}_{dataset}_B"
        f"{batch_size * gradient_accumulation_steps}.pt")
    best_model_path = save_path[:-3] + "_BEST.pt"

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
    train_dataloader, eval_dataloader = get_dataloaders(chip_ids, dataset, batch_size=batch_size)

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
            state_dict = torch.load(saved_state_path)  # , map_location=accelerator.device)
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
        for b, batch in enumerate(tqdm(train_dataloader)):
            # with accelerator.accumulate(model):
            inputs, targets, _ = batch
            outputs = model(inputs.to(device))
            loss = loss_function(outputs, targets.to(device))
            # accelerator.backward(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if b == 2:
                break

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
        for b, batch in enumerate(tqdm(eval_dataloader)):
            inputs, targets, _ = batch
            with torch.no_grad():
                predictions = model(inputs.to(device))
            # Gather all predictions and targets
            # all_predictions, all_targets = accelerator.gather_for_metrics((predictions, targets))
            # num_elements += all_predictions.shape[0]
            # val_loss += loss_function(all_predictions, all_targets).item()
            val_loss += loss_function(predictions, targets.to(device)).item()
            if b == 2:
                break

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

if __name__ == "__main__":
    dataset = "Sentinel-2all"
    mixed_precision = "fp16"
    seed = 123
    batch_size = 8
    gradient_accumulation_steps = 4
    nb_epochs = 1
    train_mode = "tune"

    train_args = (dataset, mixed_precision, seed, batch_size,
                  gradient_accumulation_steps, nb_epochs)
    training_loop(*train_args)