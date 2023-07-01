import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

GROUP_SIZE = 2

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, input_size, output_size):
        """Depthwise Separable Convolution Block
        Args:
            input_size (int): Input Channel Size
            output_size (int): Output Channel Size
        """
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(input_size, input_size, kernel_size=3, padding=1, groups=input_size)
        self.pointwise = nn.Conv2d(input_size, output_size, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class Block(nn.Module):
    def __init__(self, input_size, output_size, padding=1, norm='bn', usepool=True, stride=1, dilation=False, depthwise_separable_conv=False):
        """Initialize Block

        Args:
            input_size (int): Input Channel Size
            output_size (int): Output Channel Size
            padding (int, optional): Padding to be used for convolution layer. Defaults to 1.
            norm (str, optional): Type of normalization to be used. Allowed values ['bn', 'gn', 'ln']. Defaults to 'bn'.
            usepool (bool, optional): Enable/Disable Maxpolling. Defaults to True.
            stride (int, optional): Controls the stride for the cross-correlation. Defaults to 1.
            dilation (bool, optional): Enable/Disable dilation effect. Defaults to False
            depthwise_separable_conv (bool, optional): Enable/Disable Depthwise Separable Convolution effect. Defaults to False
        """
        super(Block, self).__init__()
        self.usepool = usepool
        temp_output_size =int(output_size/4)
        self.conv1 = nn.Conv2d(input_size, temp_output_size, 3, padding=padding)
        if norm == 'bn':
            self.n1 = nn.BatchNorm2d(temp_output_size)
        elif norm == 'gn':
            self.n1 = nn.GroupNorm(GROUP_SIZE, temp_output_size)
        elif norm == 'ln':
            self.n1 = nn.GroupNorm(1, temp_output_size)

        if dilation:
            self.conv2 = nn.Conv2d(temp_output_size, (temp_output_size)*2, 3, padding=padding, dilation=2)
        elif depthwise_separable_conv:
            self.conv2 = DepthwiseSeparableConv(temp_output_size, (temp_output_size)*2)
        else:
            self.conv2 = nn.Conv2d(temp_output_size, (temp_output_size)*2, 3, padding=padding)

        if norm == 'bn':
            self.n2 = nn.BatchNorm2d((temp_output_size)*2)
        elif norm == 'gn':
            self.n2 = nn.GroupNorm(GROUP_SIZE, (temp_output_size)*2)
        elif norm == 'ln':
            self.n2 = nn.GroupNorm(1, (temp_output_size)*2)
        # The third Conv layer of each block should have stride of 2.
        self.conv3 = nn.Conv2d((temp_output_size)*2, output_size, 3, padding=padding, stride=stride)
        if norm == 'bn':
            self.n3 = nn.BatchNorm2d(output_size)
        elif norm == 'gn':
            self.n3 = nn.GroupNorm(GROUP_SIZE, output_size)
        elif norm == 'ln':
            self.n3 = nn.GroupNorm(1, output_size)
        if usepool:
            self.pool = nn.MaxPool2d(2, 2)

    def __call__(self, x):
        """
        Args:
            x (tensor): Input tensor to this block

        Returns:
            tensor: Return processed tensor
        """
        x = self.conv1(x)
        x = self.n1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.n2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.n3(x)
        x = F.relu(x)
        if self.usepool:
            x = self.pool(x)
        return x


class Net(nn.Module):
    """ Network Class

    Args:
        nn (nn.Module): Instance of pytorch Module
    """

    def __init__(self, base_channels=32, drop=0.01, norm='bn'):
        """Initialize Network

        Args:
            base_channels (int, optional): Number of base channels to start with. Defaults to 4.
            drop (float, optional): Dropout value. Defaults to 0.01.
            norm (str, optional): Normalization type. Defaults to 'bn'.
        """
        super(Net, self).__init__()

        self.base_channels = base_channels
        self.drop = drop

        # Conv
        self.block1 = Block(3, self.base_channels, norm=norm, usepool=False)
        self.dropout1 = nn.Dropout(self.drop)
        self.block2 = Block(self.base_channels,
                            self.base_channels, norm=norm, usepool=False,
                            dilation=True)
        self.dropout2 = nn.Dropout(self.drop)
        self.block3 = Block(self.base_channels,
                            self.base_channels, norm=norm, usepool=False,
                            depthwise_separable_conv=True)
        self.dropout3 = nn.Dropout(self.drop)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flat = nn.Conv2d(self.base_channels, 10, 1)

    def forward(self, x, dropout=True):
        """Convolution function

        Args:
            x (tensor): Input image tensor
            dropout (bool, optional): Enable/Disable Dropout. Defaults to True.

        Returns:
            tensor: tensor of logits
        """
        # Conv Layer
        x = self.block1(x)
        if dropout:
            x = self.dropout1(x)
        x = self.block2(x)
        if dropout:
            x = self.dropout2(x)
        x = self.block3(x)

        # Output Layer
        x = self.gap(x)
        x = self.flat(x)
        x = x.view(-1, 10)

        # Output Layer
        return F.log_softmax(x, dim=1)
