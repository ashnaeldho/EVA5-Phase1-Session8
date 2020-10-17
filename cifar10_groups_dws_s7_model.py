import torch.nn as nn


class S7_CIFAR10(nn.Module):
    """
        Info: The model for CIFAR10 data set session 7.
        Target: To achieve more than 80% test accuracy with less than 1 million parameters, mandatory to use depth wise separable convolution, Global Average Pooling.
        """

    def __init__(self):
        super(S7_CIFAR10, self).__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False, groups=4),
            nn.BatchNorm2d(32))

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False, groups=8),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False, groups=16),
            nn.BatchNorm2d(128))

        self.pool = nn.MaxPool2d(2, 2)

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False, dilation=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False, groups=32),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, groups=256, padding=1, bias=False),
            nn.Conv2d(256, 512, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(512))

        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, padding=0, bias=False, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, groups=256, padding=1, bias=False),
            nn.Conv2d(256, 256, kernel_size=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 64, 1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            # nn.Conv2d(64, 64, kernel_size=3, groups=64, padding=0, bias=False),
            nn.Conv2d(64, 10, 1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.pool(x)
        x = self.conv_block_3(x)
        # x = self.pool(x)
        x = self.conv_block_4(x)

        x = self.gap(x)

        x = x.view(-1, 10)

        return x
