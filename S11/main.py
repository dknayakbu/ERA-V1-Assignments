import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import numpy as np
import albumentations as A
import torch
from torch.utils.data import Dataset

from models.model import *
from utils import *

LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-4
SCH_MAX_LR = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 20
model = ResNet18().to(device)

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


# Dataset and Creating Train/Test Split
train_set = CustomDataset(
            datasets.CIFAR10("./data", train=True, download=True),
            transforms=train_transforms,
        )
test_set = CustomDataset(
            datasets.CIFAR10("./data", train=False, download=True),
            transforms=test_transforms,
        )

SEED = 1
# CUDA?
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)

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

# Define the loss function, optimizer and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = SCH_MAX_LR, 
                                                    steps_per_epoch = len(train_loader), epochs = 30, 
                                                    pct_start = 5/30, div_factor = 100, three_phase = False,
                                                    final_div_factor = 100, anneal_strategy = 'linear')
