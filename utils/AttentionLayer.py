import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(AttentionLayer, self).__init__()
        self.chanel_in = in_dim
        # self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        # query = np.load(memory_path)
        # self.proj_query = torch.from_numpy(query).unsqueeze(0).cuda()
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x, y):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = y.view(m_batchsize, -1, width * height).permute(0, 2, 1).float()  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height).float()  # B X C x (*W*H)
        energy = torch.bmm(proj_key, proj_query)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(attention.permute(0, 2, 1), proj_value)
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out
