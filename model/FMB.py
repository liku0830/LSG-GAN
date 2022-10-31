import torch
import torch.nn as nn

class FMB(nn.Module):                # include template information and texture information
    def __init__(self):
        super(FMB, self).__init__()

        self.stage1 = nn.Sequential(                 # 1/8      template+mask
            nn.Conv2d(257, 512, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.stage2 = nn.Sequential(                  # 1/8     template+mask+pose
            nn.Conv2d(256+256, 512, 3, padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.Output_stage = nn.Sequential(                       # 1/8     template+mask+pose -> ground truth
            nn.Conv2d(512, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
        )
        self.stage3 = nn.Sequential(                        # 1/8
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.stage4 = nn.Sequential(                     # 1/4
            nn.ConvTranspose2d(512, 256, kernel_size=(2, 2), padding=(0, 0), stride=2),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.stage5 = nn.Sequential(                      # 1/2
            nn.ConvTranspose2d(256, 128, kernel_size=(2, 2), padding=(0, 0), stride=2),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

    def forward(self, x, m, p):        # x -> image, m -> mask, p -> pose
        # x = self.encoder(x)         # x_1 -> 1/2, x_2 -> 1/4, x_3 -> 1/8
        t_m = torch.cat((x, m), dim=1)        # t_m -> x_3+mask
        t_m = self.stage1(t_m)
        t_m_p = torch.cat((t_m, p), dim=1)       # t_m_p -> x_3+mask+pose
        t_m_p = self.stage2(t_m_p)
        t_out = self.Output_stage(t_m_p)  # t_out project to template ground truth
        t_1_8 = self.stage3(t_m_p)  # t_1  -> 1/8
        t_1_4 = self.stage4(t_1_8)      # t_2  -> 1/4
        t_1_2 = self.stage5(t_1_4)     # t_3  -> 1/2
        return t_1_2, t_1_4, t_1_8, t_out