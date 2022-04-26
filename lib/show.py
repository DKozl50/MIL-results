import matplotlib.pyplot as plt
import numpy as np
from IPython import display
from torchvision import transforms


def show_img_with_mask(img, mask, figsize=(14, 8)):
    """Shows image and mask.

    Parameters
    ----------
    img : np.ndarray
        Image.
    mask : np.ndarray
        Mask.
    figsize : tuple of 2 int, optional (default=(14, 8))
        Figure size.

    """
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.imshow(img)
    ax2.imshow(mask)
    ax1.axis("off")
    ax2.axis("off")
    plt.show()

    
def show_img_with_mask_torch(img, mask, figsize=(14, 8)):
    """
    Shows image and mask. Works for pytorch format inputs.

    Parameters
    ----------
    img : torch.tensor
        Image.
    mask : torch.tensor
        Mask.
    figsize : tuple of 2 int, optional (default=(14, 8))
        Figure size.

    """
    img = transforms.ToPILImage()(img)
    mask = transforms.ToPILImage()(mask)
    show_img_with_mask(img, mask, figsize)
    

def show_predictions_bulk_torch(imgs, preds, masks, grid=(3, 2), figsize=(14, 8)):
    """Shows images, predictions and masks.

    Parameters
    ----------
    imgs : torch.tensor
        Images.
    preds : torch.tensor
        binarized predictions
    masks : torch.tensor
        Masks.
    grid : tuple of 2 int, optional (default=(3, 2))
        How to stack multiple image triplets
    figsize : tuple of 2 int, optional (default=(14, 8))
        Figure size.

    """
    grid_v, grid_h = grid
    requested_pics = grid_v * grid_h
    pic_indices = np.random.choice(len(imgs), requested_pics, replace=False)
    pic_iter = iter(pic_indices)
    grid_h *= 3
    f, axs = plt.subplots(grid_v, grid_h, figsize=figsize)
    for i in range(0, len(axs[0]), 3):
        axs[0][i].set_title('Image')
        axs[0][i+1].set_title('Predicted')
        axs[0][i+2].set_title('True Mask')
        
    for row in axs:
        for i in range(0, len(row), 3):
            pic_index = next(pic_iter)
            img = transforms.ToPILImage()(imgs[pic_index])
            pred = transforms.ToPILImage()(preds[pic_index])
            mask = transforms.ToPILImage()(masks[pic_index])
            row[i].imshow(img)
            row[i+1].imshow(pred)
            row[i+2].imshow(mask)
            row[i].axis('off')
            row[i+1].axis('off')
            row[i+2].axis('off')
    plt.tight_layout(pad=0.4)
    plt.show()


def plot_losses(train_losses, val_losses, epoch, plot_output=None):
    """
    Plots losses as well as current training info.
    First 5 loss values are omitted if possible, because they tend to be huge.

    Parameters
    ----------
    train_losses : list
        Training losses. Length must be equal to ``epoch``.
    val_losses : list
        Validation losses. Length must be equal to ``epoch``.
    epoch : int
        How many epochs have passed.
    plot_output : ipywidgets Output, optional
        Output object to redraw cell output.
        If not provided, function will append it's plot to current cell output.

    Returns
    -------
    None

    """
    if plot_output is not None:
        with plot_output:
            _plot_losses(train_losses, val_losses, epoch)
            display.clear_output(wait=True)
    else:
        _plot_losses(train_losses, val_losses, epoch)


def _plot_losses(ta, va, epoch):
    """Internal function for ``plot_losses``. Refer to if for documentation."""
    # get rid of first epochs with insane losses
    starting_point = min(epoch - 5, 5)
    starting_point = max(0, starting_point)
    ta = ta[starting_point:]
    va = va[starting_point:]
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('epochs')
    ax1.plot(np.arange(starting_point, epoch), ta, color='b', label='train loss')
    ax1.plot(np.arange(starting_point, epoch), va, color='r', label='val loss')
    ax1.legend()
    plt.title(f'Epoch {epoch} | Val Loss = {va[-1]}')
    plt.grid()
    plt.show()
