import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn.functional as F

# Train Phase transformations
# Define the Albumentations transformations
train_transforms = A.Compose([
    A.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)), # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values.
    A.PadIfNeeded(min_height=40,min_width=40),
    A.RandomCrop(32, 32, p=1.0),
    A.HorizontalFlip(),
    A.Cutout(num_holes=1, max_h_size=8, max_w_size=8, fill_value=[0.4914, 0.4822, 0.4465], always_apply=False, p=0.1),
    ToTensorV2(),
])

# Test Phase transformations
test_transforms = A.Compose([
    A.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)), # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values.
    ToTensorV2()
])

import matplotlib.pyplot as plt
def print_samples(loader, class_map, count=16):
    """Print samples input images

    Args:
        loader (DataLoader): dataloader for training data
        count (int, optional): Number of samples to print. Defaults to 16.
    """
    # Print Random Samples
    if not count % 8 == 0:
        return

    classes = list(class_map.keys())
    fig = plt.figure(figsize=(15, 5))
    for imgs, labels in loader:
        for i in range(count):
            ax = fig.add_subplot(int(count / 8), 8, i + 1, xticks=[], yticks=[])
            ax.set_title(f"{classes[labels[i]]}")
            plt.imshow(imgs[i].cpu().numpy().transpose(1, 2, 0).astype(float))
        break


def train(model, device, train_loader, optimizer, epoch):
  train_losses = []
  train_acc = []

  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = F.nll_loss(y_pred, target)
    train_losses = (loss)

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Update pbar-tqdm

    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_acc = (100*correct/processed)

    return train_losses, train_acc


def test(model, device, test_loader):

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses = (test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_acc = (100. * correct / len(test_loader.dataset))

    return test_losses, test_acc


def progress_bar(EPOCHS, model, device, train_loader, test_loader, optimizer, lr_scheduler):
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []

    for epoch in range(EPOCHS):
        print("EPOCH:", epoch)
        train_losses_epoch, train_acc_epoch = train(model, device, train_loader, optimizer, epoch)
        train_losses.append(train_losses_epoch)
        train_acc.append(train_acc_epoch)
        lr_scheduler.step()
        test_losses_epoch, test_acc_epoch = test(model, device, test_loader)
        test_losses.append(test_losses_epoch)
        test_acc.append(test_acc_epoch)

    return train_losses, test_losses, train_acc, test_acc


def get_incorrrect_predictions(model, loader, device):
    """Get all incorrect predictions

    Args:
        model (Net): Trained model
        loader (DataLoader): instance of data loader
        device (str): Which device to use cuda/cpu

    Returns:
        list: list of all incorrect predictions and their corresponding details
    """
    model.eval()
    incorrect = []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target)
            pred = output.argmax(dim=1)
            for d, t, p, o in zip(data, target, pred, output):
                if p.eq(t.view_as(p)).item() == False:
                    incorrect.append(
                        [d.cpu(), t.cpu(), p.cpu(), o[p.item()].cpu()])

    return incorrect


def plot_incorrect_predictions(predictions, class_map, count=10):
    """Plot Incorrect predictions

    Args:
        predictions (list): List of all incorrect predictions
        class_map (dict): Lable mapping
        count (int, optional): Number of samples to print, multiple of 5. Defaults to 10.
    """
    print(f'Total Incorrect Predictions {len(predictions)}')

    if not count % 5 == 0:
        print("Count should be multiple of 10")
        return

    classes = list(class_map.keys())

    fig = plt.figure(figsize=(10, 5))
    for i, (d, t, p, o) in enumerate(predictions):
        ax = fig.add_subplot(int(count/5), 5, i + 1, xticks=[], yticks=[])
        ax.set_title(f'{classes[t.item()]}/{classes[p.item()]}')
        show_image = d.cpu().numpy().transpose(1, 2, 0)
        # Rescale the pixel values to [0, 1]
        show_image = (show_image - np.min(show_image)) / (np.max(show_image) - np.min(show_image))
        plt.imshow(show_image)
        if i+1 == 5*(count/5):
            break