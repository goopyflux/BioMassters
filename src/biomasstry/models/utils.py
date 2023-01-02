"""Module for commonly shared functions for training a model."""

from time import time

import numpy as np
import torch
from tqdm import tqdm

# if torch.cuda.is_available():
    # device = torch.device('cuda')
# else:
    # device = torch.device('cpu')
# print(device)

# Train and Validation Loops
def train_loop(dataloader, model, loss_fn, optimizer):
    train_metrics = []
    
    print('Training')
    for ix, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        X = batch['image']  # .to(device)
        y = batch['target']  # .to(device)
        
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_metrics.append(np.round(np.sqrt(loss.item()), 5))
            
    return train_metrics

def valid_loop(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    valid_loss = 0
    valid_metrics = {}

    print('Validation')
    with torch.no_grad():
        for batch in tqdm(dataloader, total=num_batches):
            X = batch['image']  # .to(device)
            y = batch['target']  # .to(device)
            
            pred = model(X)
            valid_loss += loss_fn(pred, y).item()
            
    valid_loss /= num_batches
    valid_rmse = np.round(np.sqrt(valid_loss), 5)
    print(f"Validation Error: \n RMSE: {valid_rmse:>8f} \n")
    return valid_rmse

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
