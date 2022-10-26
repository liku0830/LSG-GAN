import torch
import torch.nn as nn



class GlobalFlow(nn.Module):
    def __init__(self, input_channel):
        super(GlobalFlow, self).__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(input_channel, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv_block_2 = nn.Sequential(
            nn.AvgPool2d(2, stride=2, ceil_mode=True),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2, ceil_mode=True)
        )
        self.conv_block_3 = nn.Sequential(
            nn.AvgPool2d(2, stride=2, ceil_mode=True),
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conv_block_4 = nn.Sequential(
            nn.AvgPool2d(2, stride=2, ceil_mode=True),
            nn.Conv2d(256, 512, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.conv_block_5 = nn.Sequential(  # 1/16
            nn.AvgPool2d(2, stride=2, ceil_mode=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, kernel_size=(2, 2), padding=(0, 0), stride=2),
        )

    def forward(self, x, p1, p2):
        x = torch.cat((x, p1, p2), dim=1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        x = self.conv_block_5(x)
        return x