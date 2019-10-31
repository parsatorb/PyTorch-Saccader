"""
https://arxiv.org/pdf/1908.07644.pdf
"""

import torch.nn as nn

from Bagnet import Bagnet77
from AttentionCell import AttentionCell
from SaccaderCell import SaccaderCell

class Saccader(nn.Module):
    """
    The whole saccader network assembled
    """

    def __init__(self, num_classes=1000):
        super().__init__()

        self.representation_network = Bagnet77()
        self.attention_network = AttentionCell(in_channels=1024)
        self.saccader_cell = SaccaderCell()

        self.conv1 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(512, num_classes, kernal_size=1, stride=1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(1024, 512, kernal_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        what = self.representation_network(x)
        where = self.attention_network(what)

        what = self.conv1(what)

        where = torch.cat([what, where], 1)
        where = self.conv3(where)

        what = self.conv2(what)
        where = self.saccader_cell(where)

        ## TODO: Attention policy
        # out = slice(attention, what)

        return out
