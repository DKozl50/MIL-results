from IPython import display
from ipywidgets import Output

import torch
from torch import nn
from tqdm.auto import tqdm

from .metrics import get_dice_torch
from .show import plot_losses, show_predictions_bulk_torch


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains model for one epoch.

    Parameters
    ----------
    model : nn.Module
        Trainable model already on device.
    dataloader : DataLoader
        DataLoader, usually for train dataset.
    criterion : Callable
        Differentiable torch loss. It will be minimized.
    optimizer : Any
        Torch optimizer.
    device : torch.device
        Device. Usually ``torch.device('cpu')`` or ``torch.device('cuda')``.

    Returns
    -------
    float
        Mean loss across all batches.
    """
    model = model.train()
    all_losses = []
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        predictions = model(images)
        loss = criterion(predictions, masks)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        all_losses.append(loss.cpu().item())
    return sum(all_losses)/len(all_losses)


def predict_val(model, dataloader, criterion, device):
    """
    Evaluate model and predict.

    Parameters
    ----------
    model : nn.Module
        Model already on device.
    dataloader : DataLoader
        DataLoader, usually for validation dataset.
    criterion : Callable
        Loss criterion. It does not need to be differentiable.
    device : torch.device
        Device. Usually ``torch.device('cpu')`` or ``torch.device('cuda')``.

    Returns
    -------
    float
        Mean loss across all batches.
    torch.tensor, 4d
        Predictions for all elements of passed dataloader. [B, C, H, W]
    torch.tensor, 4d
        True masks for all elements of passed dataloader. [B, C, H, W]
    """
    model = model.eval()
    with torch.no_grad():
        all_losses = []
        preds_all = []
        masks_all = []
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            predictions = model(images)
            loss = criterion(predictions, masks).item()
            all_losses.append(loss)
            predictions = nn.Sigmoid()(predictions).round().cpu()
            masks = masks.cpu()
            preds_all.append(predictions)
            masks_all.append(masks)
        return sum(all_losses)/len(all_losses), preds_all, masks_all

    
def predict_test(model, dataloader, device):
    """
    Evaluate model and predict.

    Parameters
    ----------
    model : nn.Module
        Model already on device.
    dataloader : DataLoader
        DataLoader, usually for test dataset (without masks available).
    device : torch.device
        Device. Usually ``torch.device('cpu')`` or ``torch.device('cuda')``.

    Returns
    -------
    torch.tensor, 4d
        Predictions for all elements of passed dataloader. [B, C, H, W]
    """
    model = model.eval()
    with torch.no_grad():
        preds_all = []
        for images in dataloader:
            images = images.to(device)
            predictions = model(images)
            predictions = nn.Sigmoid()(predictions).round().cpu()
            preds_all.append(predictions)
        return preds_all

    
def train(model, train_dataloader, val_dataloader, val_dataset, criterion, optimizer, device, n_epochs=50):
    """
    Train model.

    Parameters
    ----------
    model : nn.Module
        Trainable model already on device.
    train_dataloader : DataLoader
        DataLoader with training data.
    val_dataloader : DataLoader
        DataLoader with training data.
    val_dataset : Dataset
        Dataset with validation data, used for visualization.
    criterion : Callable
        Differentiable torch loss. It will be minimized.
    optimizer : Any
        Torch optimizer.
    device : torch.device
        Device. Usually ``torch.device('cpu')`` or ``torch.device('cuda')``.
    n_epochs : int, optional (default=50)
        How many epochs to train for.

    Returns
    -------
    None
    """
    plot_output = Output()
    display.display(plot_output)
    
    train_losses = []
    val_losses = []
    with tqdm(total=n_epochs) as progress:
        for epoch in range(n_epochs):
            train_loss = train_epoch(model, train_dataloader, criterion, optimizer, device)
            train_losses.append(train_loss)
            val_loss, val_preds, val_masks = predict_val(model, val_dataloader, criterion, device)
            val_preds = torch.cat(val_preds, dim=0)
            val_masks = torch.cat(val_masks, dim=0)
            val_losses.append(val_loss)
            progress.update(1)
            progress.set_description(f'dice: {get_dice_torch(val_masks, val_preds)} ')
            with plot_output:
                plot_losses(train_losses, val_losses, epoch+1)
                show_predictions_bulk_torch(val_dataset.images, val_preds, val_dataset.masks, figsize=(10, 5))
                display.clear_output(wait=True)


def soft_dice_loss(mask_pred, mask_true):
    """
    Differentiable Dice score approximation.
    For our task it should work better than BCE.
    https://arxiv.org/pdf/1911.01685.pdf

    For batch input returns an average loss.

    Parameters
    ----------
    mask_pred : torch.tensor, 4d
        Predicted mask logits. [B, C, H, W]
    mask_true : torch.tensor, 4d
        Ground truth mask. [B, C, H, W]

    Returns
    -------
    loss : torch.tensor, 1d
        Computed loss.
    """
    mask_pred = nn.Sigmoid()(mask_pred)
    numerator = 2 * (mask_true * mask_pred).sum(dim=[2, 3])
    denominator = (mask_true + mask_pred).sum(dim=[2, 3])
    return 1-(numerator/denominator).mean()
