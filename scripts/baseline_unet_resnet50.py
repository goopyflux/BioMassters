# %% [markdown]
# # Train Baseline UNET model with Single Satellite Image

# %%
!pip install --upgrade rasterio s3fs

# %%
# Install the local package
!pip install -e /notebooks/

# %%
import os
from time import time

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm.notebook import tqdm

# %%
sns.set()

# %%
%load_ext autoreload
%autoreload 2

# %%
from biomasstry.datasets import Sentinel2

# %%
sen2dataset = Sentinel2()

# %%
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(device)

# %%
# split
torch.manual_seed(0)
train_frac = 0.8
train_samples = round(train_frac * len(sen2dataset))
val_samples = round((1 - train_frac) * len(sen2dataset))

train_dataset, val_dataset = random_split(sen2dataset, [train_samples, val_samples])
print(f"Train samples: {len(train_dataset)} "
      f"Val. samples: {len(val_dataset)}")

# %%
# Model
img_data = train_dataset[0]['image']
in_channels = img_data.shape[0]
print(f'# input channels: {in_channels}')
print(f"Image shape: {img_data.shape}")

model = smp.Unet(
    encoder_name="resnet50",
    encoder_weights=None, # 'imagenet' weights don't seem to help so start clean 
    in_channels=in_channels,                 
    classes=1,                     
).to(device)

# %%
# Loss and Optimizer
loss_module = nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

# %%
# Train and Validation Loops
def train_loop(dataloader, model, loss_fn, optimizer):
    train_metrics = []
    
    print('Training')
    for ix, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        X = batch['image'].to(device)
        y = batch['target'].to(device)
        
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_metrics.append(np.round(np.sqrt(loss.item()), 5))
            
    return train_metrics

# %%
def valid_loop(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    valid_loss = 0
    valid_metrics = {}

    print('Validation')
    with torch.no_grad():
        for batch in tqdm(dataloader, total=num_batches):
            X = batch['image'].to(device)
            y = batch['target'].to(device)
            
            pred = model(X)
            valid_loss += loss_fn(pred, y).item()
            
    valid_loss /= num_batches
    valid_rmse = np.round(np.sqrt(valid_loss), 5)
    print(f"Validation Error: \n RMSE: {valid_rmse:>8f} \n")
    return valid_rmse

# %%
def run_training(model, loss_module, optimizer, train_dataloader, val_dataloader, save_path, n_epochs=10):
    min_valid_metric = np.inf
    train_metrics = []
    valid_metrics = []
    
    total_train_time = 0
    total_val_time = 0

    for ix in range(n_epochs):
        print(f"\n-------------------------------\nEpoch {ix+1}")
        start = time()
        train_metrics_epoch = train_loop(train_dataloader, model, loss_module, optimizer)
        end = time()
        train_time = end - start
        total_train_time += train_time
        train_metrics.extend(train_metrics_epoch)
        
        start = time()
        valid_metrics_epoch = valid_loop(val_dataloader, model, loss_module)
        end = time()
        val_time = end - start
        total_val_time += val_time
        valid_metrics.append((len(train_metrics), valid_metrics_epoch))

        # check validation score, if improved then save model
        if min_valid_metric > valid_metrics_epoch:
            print(f'Validation RMSE Decreased({min_valid_metric:.6f}--->{valid_metrics_epoch:.6f}) \t Saving The Model')
            min_valid_metric = valid_metrics_epoch

            # Saving State Dict
            torch.save(model.state_dict(), save_path)
        print(f"Train time: {train_time}. Validation time: {val_time}")
    print("Done!")
    print(f"Total train time: {total_train_time} s. Avg. time per epoch: {total_train_time / n_epochs}")
    print(f"Total val time: {total_val_time} s. Avg. time per epoch: {total_val_time / n_epochs}")
    train_metrics_zipped = list(zip(np.arange(0, len(train_metrics)), train_metrics))
    
    return {'training': train_metrics_zipped, 'validation': valid_metrics}

# %% [markdown]
# ## Experiment with `num_workers` and `batch_size` for tuning `DataLoader` Throughput

# %%
# DataLoaders
# num_workers = 4
# batch_size = 64  # Note: training speed is sensitive to memory usage
                 # set this as high as you can without significantly slowing down training time 

dir_saved_models = "../artifacts"
# Expt. with num_workers and batch_size
timing = []
for batch_size in [64]:
    print(f"Batch size: {batch_size}")
    for num_workers in [6]:
        print(f"Number of workers = {num_workers}")
        train_dataloader = DataLoader(train_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=num_workers,
                                    pin_memory=True
                                    )

        val_dataloader = DataLoader(val_dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=num_workers,
                                    pin_memory=True
                                )

        save_file = f"UNET_resnet50_10bandS2Apr_batch_AGBMLinear_1epoch_08DEC.pt"
        save_path = os.path.join(dir_saved_models, save_file)
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
        # epoch_time = time() - start
        # timing.append((num_workers, batch_size, epoch_time))
        # print(f"time for one epoch = {epoch_time}")

# %%
print("Workers  Batch  Time")
for t in timing:
    print(f"{t[0]}        {t[1]}     {t[2]}")

# %%
timing_df = pd.DataFrame(timing, columns=["workers", "batch", "time"])

# %%
timing_df.to_csv(f"../artifacts/worker_{num_workers}_batch_{batch_size}_timing.csv", index=False)

# %%
# sns.catplot(data=timing_df, x="workers", y="time", hue="batch", kind="bar")


