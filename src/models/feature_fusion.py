import torch
import torch.nn as nn


# SE MODEL
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c, 1, 1)
        y = self.fc(y)
        return x * y.expand_as(x)

class FF(nn.Module):
    #Feature Fusion
    def __init__(self, origin_channels, in_channels, out_channels):
        super(FF, self).__init__()
        self.conv_dim = nn.Conv2d(origin_channels, in_channels, 3, 1, 1)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.conv_pre = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        
        self.fusion_dot = nn.Sequential(
            nn.Conv2d(in_channels=in_channels + in_channels, out_channels=out_channels, kernel_size=3, stride=1,
                      padding=1),
            SELayer(out_channels),
            nn.Sigmoid()
        )

        # self.fusion_add = nn.Sequential(
        #     nn.Conv2d(in_channels=in_channels + in_channels, out_channels=out_channels, kernel_size=3, stride=1,
        #               padding=1),
        #     SELayer(out_channels),
        #     nn.Sigmoid()
        # )

        # self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, x_skip, x_next=None):
        if self.conv_dim is not None:
            x = self.conv_dim(x)
            x = self.activation(x)
            x = self.conv_pre(x)
        
        cat = torch.cat((x, x_skip), dim=1)
        # print("cat.shape:", cat.shape)

        map_dot = self.fusion_dot(cat)
        # map_add = self.fusion_add(cat)
        # ff_out = self.gamma * (map_cat * x_next)

        return map_dot, x