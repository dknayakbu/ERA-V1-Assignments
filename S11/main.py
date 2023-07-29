import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import numpy as np
import albumentations as A
import torch
from torch.utils.data import Dataset
from torchsummary import summary

from models.model import *
from utils import *

LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
SCH_MAX_LR = 1e-2
SEED = 1
EPOCHS = 20
##############################################################################
# CUDA?
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

##############################################################################
class CustomDataset(Dataset):
    """
    Custom Dataset Class
    """
    def __init__(self, dataset, transforms = None):
        """Initialize Dataset
        Args:
            dataset (Dataset): Pytorch Dataset instance
            transforms (Transform.Compose, optional): Tranform function instance. Defaults to None.
        """
        self.transforms = transforms
        self.dataset = dataset

    def __len__(self):
        """Get dataset length
        Returns:
            int: Length of dataset
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """Get an item form dataset
        Args:
            idx (int): id of item in dataset
        Returns:
            (tensor, int): Return tensor of transformer image, label
        """
        # Read Image and Label
        image, label = self.dataset[idx]
        image = np.array(image)

        # Apply Transforms
        if self.transforms is not None:
            image = self.transforms(image=image)["image"]

        return (image, label)

##############################################################################
# Dataset and Creating Train/Test Split
train_set = CustomDataset(
            datasets.CIFAR10("./data", train=True, download=True),
            transforms=train_transforms,
        )
test_set = CustomDataset(
            datasets.CIFAR10("./data", train=False, download=True),
            transforms=test_transforms,
        )

class_map = {
    "PLANE": 0,
    "CAR": 1,
    "BIRD": 2,
    "CAT": 3,
    "DEER": 4,
    "DOG": 5,
    "FROG": 6,
    "HORSE": 7,
    "SHIP": 8,
    "TRUCK": 9,
}
##############################################################################
# For reproducibility
torch.manual_seed(SEED)

if cuda:
    torch.cuda.manual_seed(SEED)

# dataloader arguments - something you'll fetch these from cmdprmt
dataloader_args = dict(shuffle=True, batch_size=128, num_workers=2, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

# train dataloader
train_loader = torch.utils.data.DataLoader(train_set, **dataloader_args)

# test dataloader
test_loader = torch.utils.data.DataLoader(test_set, **dataloader_args)

# print_samples(train_loader, class_map)
##############################################################################
# Model summary details
model = ResNet18().to(device)
# print(summary(model, input_size=(3, 32, 32)))
##############################################################################
# Define the loss function, optimizer and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = SCH_MAX_LR, 
#                                                     steps_per_epoch = len(train_loader), epochs = EPOCHS, 
#                                                     pct_start = 5/EPOCHS, div_factor = 100, three_phase = False,
#                                                     final_div_factor = 100, anneal_strategy = 'linear')
##############################################################################
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
        train_losses.append(loss)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Update pbar-tqdm

        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        train_acc.append(100*correct/processed)
    return train_losses, train_acc

def test(model, device, test_loader):
    test_losses = []
    test_acc = []

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
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_acc.append(100. * correct / len(test_loader.dataset))
    return test_losses, test_acc
