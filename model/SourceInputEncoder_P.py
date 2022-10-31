import torch
import torch.nn as nn



class PoseFlow(nn.Module):
    def __init__(self):
        super(PoseFlow, self).__init__()
        self.pose_block_1 = nn.Sequential(
            nn.Conv2d(18, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.pose_block_2 = nn.Sequential(  # 1/2
            nn.AvgPool2d(2, stride=2, ceil_mode=True),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.pose_block_3 = nn.Sequential(  # 1/4
            nn.AvgPool2d(2, stride=2, ceil_mode=True),
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.pose_block_4 = nn.Sequential(  # 1/8
            nn.AvgPool2d(2, stride=2, ceil_mode=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )


    def forward(self, x):
        x = self.pose_block_1(x)
        p_1_2 = self.pose_block_2(x)
        p_1_4 = self.pose_block_3(p_1_2)
        p_1_8 = self.pose_block_4(p_1_4)

        return p_1_2, p_1_4, p_1_8