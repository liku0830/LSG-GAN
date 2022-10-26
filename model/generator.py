import torch
import torch.nn as nn
from utils.AttentionLayer import *
from torch.nn import init
from globalflow import GlobalFlow
from imageEncoder import ImageEncoder
from templateflow import TemplateFlow
from poseflow import PoseFlow


class MainFlow(nn.Module):
    def __init__(self):
        super(MainFlow, self).__init__()
        self.g_flow = GlobalFlow(3+36)
        self.i_flow = ImageEncoder()
        self.t_flow = TemplateFlow()
        self.p_flow = PoseFlow()


        self.atten_template_8 = AttentionLayer(512)
        self.atten_texture_8 = AttentionLayer(512)
        self.atten_template_4 = AttentionLayer(256)
        self.atten_texture_4 = AttentionLayer(256)
        self.atten_template_2 = AttentionLayer(128)
        self.atten_texture_2 = AttentionLayer(128)


        self.texture_pose_sampling_1 = nn.Sequential(  # 1/8
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.texture_fuse1 = nn.Sequential(  # 1/8
            nn.Conv2d(512, 64, kernel_size=(3, 3), padding=(1, 1), stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 18*3, kernel_size=(1, 1), padding=(0, 0)),
            nn.Tanh()
        )

        self.texture_pose_sampling_2 = nn.Sequential(  # 1/4
            nn.Conv2d(512, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.texture_fuse2 = nn.Sequential(  # 1/4
            nn.AvgPool2d(2, 2),
            nn.Conv2d(256, 64, kernel_size=(3, 3), padding=(1, 1), stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 18*3, kernel_size=(1, 1), padding=(0, 0)),
            nn.Tanh()
        )

        self.texture_pose_sampling_3 = nn.Sequential(  # 1/16
            nn.Conv2d(256, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.texture_fuse3 = nn.Sequential(  # 1/2
            nn.AvgPool2d(4, 4),
            nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1), stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 18*3, kernel_size=(1, 1), padding=(0, 0)),
            nn.Tanh()
        )

        self.deconv_block_1 = nn.Sequential(            #1/8
            nn.ConvTranspose2d(512, 512, kernel_size=(2, 2), padding=(0, 0), stride=2),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.side_path1 = nn.Sequential(            #1/8
            nn.ConvTranspose2d(512, 512, kernel_size=(8, 8), padding=(0, 0), stride=8),
            nn.Conv2d(512, 3, kernel_size=(1, 1), padding=(0, 0), stride=1),
        )

        self.deconv_block_2 = nn.Sequential(  # 1/4
            nn.ConvTranspose2d(512, 256, kernel_size=(2, 2), padding=(0, 0), stride=2),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.side_path2 = nn.Sequential(  # 1/4
            nn.ConvTranspose2d(256, 256, kernel_size=(4, 4), padding=(0, 0), stride=4),
            nn.Conv2d(256, 3, kernel_size=(1, 1), padding=(0, 0), stride=1),
        )

        self.deconv_block_3 = nn.Sequential(  # 1/2
            nn.ConvTranspose2d(256, 128, kernel_size=(2, 2), padding=(0, 0), stride=2),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.side_path3 = nn.Sequential(  # 1/2
            nn.ConvTranspose2d(128, 128, kernel_size=(2, 2), padding=(0, 0), stride=2),
            nn.Conv2d(128, 3, kernel_size=(1, 1), padding=(0, 0), stride=1),
        )

        self.deconv_block_4 = nn.Sequential(               # 1
            nn.ConvTranspose2d(128, 64, kernel_size=(2, 2), padding=(0, 0), stride=2),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=(7, 7), padding=(3, 3)),
        )

        self.fuse = nn.Conv2d(12, 3, kernel_size=(1, 1), padding=(0, 0))

        self.activation = nn.Tanh()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, m, p1, p2, g):
        img = x
        x = self.g_flow(x, p1, p2)
        i_1_2, i_1_4, i_1_8 = self.i_flow(img)
        p_1_2, p_1_4, p_1_8 = self.p_flow(p2)
        t_1_2, t_1_4, t_1_8, t_out = self.t_flow(i_1_8, m, p_1_8)          # t_out is guided by template ground truth


        i_1_8_sampling = torch.cat((i_1_8, p_1_8), dim=1)
        i_1_8_sampling = self.texture_pose_sampling_1(i_1_8_sampling)
        i_1_8 = self.texture_fuse1(i_1_8_sampling)

        i_1_4_sampling = torch.cat((i_1_4, p_1_4), dim=1)
        i_1_4_sampling = self.texture_pose_sampling_2(i_1_4_sampling)
        i_1_4 = self.texture_fuse2(i_1_4_sampling)

        i_1_2_sampling = torch.cat((i_1_2, p_1_2), dim=1)
        i_1_2_sampling = self.texture_pose_sampling_3(i_1_2_sampling)
        i_1_2 = self.texture_fuse3(i_1_2_sampling)

        x = self.deconv_block_1(x)              # 1/8
        x = self.atten_template_8(x, t_1_8)
        s1 = self.side_path1(x)

        x = self.deconv_block_2(x)               # 1/4
        x = self.atten_template_4(x, t_1_4)
        s2 = self.side_path2(x)

        x = self.deconv_block_3(x)                # 1/2
        x = self.atten_template_2(x, t_1_2)
        s3 = self.side_path3(x)

        s4 = self.deconv_block_4(x)

        s1 = self.activation(s1)
        s2 = self.activation(s2)
        s3 = self.activation(s3)
        s4 = self.activation(s4)

        return s1, s2, s3, s4, i_1_2, i_1_4, i_1_8, t_out



