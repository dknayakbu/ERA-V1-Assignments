import torch
import torch.nn as nn

# Residual Block
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu2(out)
        return out

# Custom ResNet architecture
class CustomResNet(nn.Module):
    def __init__(self):
        super(CustomResNet, self).__init__()
        # PrepLayer
        self.prep_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Layer1
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.res_block1 = ResBlock(128, 128)

        # Layer2
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # Layer3
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.res_block2 = ResBlock(512, 512)

        # MaxPooling with Kernel Size 4
        self.maxpool = nn.MaxPool2d(kernel_size=4)

        # FC Layer
        self.fc = nn.Linear(512, 10)

        # Softmax
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.prep_layer(x)
        x = self.layer1(x)
        x = self.res_block1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.res_block2(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.softmax(x)
        return x

