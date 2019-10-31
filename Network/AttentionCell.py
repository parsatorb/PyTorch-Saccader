import torch.nn as nn

class AttentionCell(nn.Module):
    def __init__(self, in_channels):
        self.conv1 = nn.Conv2d(in_channels, in_channels,
                        kernal_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.conv2 = nn.Conv2d(in_channels, in_channels//2,
                        kernal_size=3, stride=1, padding=0, dilation=2, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels//2)

        self.conv3 = nn.Conv2d(in_channels//2, in_channels//2,
                        kernal_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(in_channels//2)

        self.conv4 = nn.Conv2d(in_channels//2, in_channels//2,
                        kernal_size=1, stride=1, padding=0, dilation=2, bias=False)
        self.bn4 = nn.BatchNorm2d(in_channels//2)

        self.relu = self.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu(out)

        return out
