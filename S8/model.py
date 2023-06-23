import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

GROUP_SIZE = 2

class Net_Batch_Norm(nn.Module):
    def __init__(self):
        super(Net_Batch_Norm, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        ) # output_size = 30

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        ) # output_size = 28

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        ) # output_size = 28

        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 14

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        ) # output_size = 12

        self.convblock5= nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        ) # output_size = 10

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        ) # output_size = 10

        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 5

        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        ) # output_size = 3

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        ) # output_size = 1

        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 1

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=1)
        ) # output_size = 1

        self.dropout = nn.Dropout(0.2)

        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 4

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = x + self.convblock3(x)
        x = self.dropout(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = x + self.convblock6(x)
        x = self.dropout(x)
        x = self.pool1(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.convblock9(x)
        x = self.gap(x)
        x = x + self.convblock10(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

class Net_Group_Norm(nn.Module):
    def __init__(self):
        super(Net_Group_Norm, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.GroupNorm(GROUP_SIZE, 16),
            nn.ReLU()
        ) # output_size = 30

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.GroupNorm(GROUP_SIZE, 16),
            nn.ReLU()
        ) # output_size = 28

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
            nn.GroupNorm(GROUP_SIZE, 16),
            nn.ReLU()
        ) # output_size = 28

        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 14

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.GroupNorm(GROUP_SIZE, 32),
            nn.ReLU()
        ) # output_size = 12

        self.convblock5= nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.GroupNorm(GROUP_SIZE, 32),
            nn.ReLU()
        ) # output_size = 10

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
            nn.GroupNorm(GROUP_SIZE, 32),
            nn.ReLU()
        ) # output_size = 10

        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 5

        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.GroupNorm(GROUP_SIZE, 32),
            nn.ReLU()
        ) # output_size = 3

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.GroupNorm(GROUP_SIZE, 32),
            nn.ReLU()
        ) # output_size = 1

        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 1

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=1)
        ) # output_size = 1

        self.dropout = nn.Dropout(0.2)

        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 4

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = x + self.convblock3(x)
        x = self.dropout(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = x + self.convblock6(x)
        x = self.dropout(x)
        x = self.pool1(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.convblock9(x)
        x = self.gap(x)
        x = x + self.convblock10(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

class Net_Layer_Norm(nn.Module):
    def __init__(self):
        super(Net_Layer_Norm, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.LayerNorm((16, 30, 30)),
            nn.ReLU()
        ) # output_size = 30

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.LayerNorm((16, 28, 28)),
            nn.ReLU()
        ) # output_size = 28

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
            nn.LayerNorm((16, 28, 28)),
            nn.ReLU()
        ) # output_size = 28

        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 14

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.LayerNorm((32, 12, 12)),
            nn.ReLU()
        ) # output_size = 12

        self.convblock5= nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.LayerNorm((32, 10, 10)),
            nn.ReLU()
        ) # output_size = 10

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
            nn.LayerNorm((32, 10, 10)),
            nn.ReLU()
        ) # output_size = 10

        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 5

        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.LayerNorm((32, 3, 3)),
            nn.ReLU()
        ) # output_size = 3

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.LayerNorm((32, 1, 1)),
            nn.ReLU()
        ) # output_size = 1

        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 1

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=1)
        ) # output_size = 1

        self.dropout = nn.Dropout(0.2)

        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 4

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = x + self.convblock3(x)
        x = self.dropout(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = x + self.convblock6(x)
        x = self.dropout(x)
        x = self.pool1(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.convblock9(x)
        x = self.gap(x)
        x = x + self.convblock10(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
