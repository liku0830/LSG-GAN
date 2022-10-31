import torch
import torch.nn as nn

class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),

        )
        self.conv_block_2 = nn.Sequential(                  # 1/2
            nn.AvgPool2d(2, stride=2, ceil_mode=True),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.conv_block_3 = nn.Sequential(                   # 1/4
            nn.AvgPool2d(2, stride=2, ceil_mode=True),
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),

        )
        self.conv_block_4 = nn.Sequential(                 # 1/8
            nn.AvgPool2d(2, stride=2, ceil_mode=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

    def forward(self, x):
        conv_1 = self.conv_block_1(x)
        conv_1_2 = self.conv_block_2(conv_1)
        conv_1_4 = self.conv_block_3(conv_1_2)
        conv_1_8 = self.conv_block_4(conv_1_4)
        return conv_1_2, conv_1_4, conv_1_8