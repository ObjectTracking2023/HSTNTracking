import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, Parameter

class SEModule(nn.Module):

    def __init__(self, channels, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

'''Non-local module'''
class DWConv(nn.Module):
    def __init__(self, in_plane, out_plane):
        super(DWConv, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_plane,out_channels=in_plane,kernel_size=1,stride=1,padding=0,groups=in_plane)
        self.point_conv = nn.Conv2d(in_channels=in_plane,out_channels=out_plane,kernel_size=1,stride=1,padding=0,groups=1)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        """
        :param in_channels:
        :param inter_channels:
        :param dimension:
        :param sub_sample:
        :param bn_layer:

        """
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 1, 1))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            conv_nd1 = nn.Conv2d
            conv_nd2 = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(1, 1))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(1))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
            self.W1 = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W1.weight, 0)
            nn.init.constant_(self.W1.bias, 0)
            self.W2 = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W2.weight, 0)
            nn.init.constant_(self.W2.bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)
            self.W1 = conv_nd1(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W1.weight, 0)
            nn.init.constant_(self.W1.bias, 0)
            self.W2 = conv_nd2(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W2.weight, 0)
            nn.init.constant_(self.W2.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        self.theta1 = conv_nd1(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi1 = conv_nd1(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        self.theta2 = conv_nd2(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi2 = conv_nd2(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        self.conv = DWConv(256 * 3, 1024)
        self.convout = DWConv(256 * 3, 1024)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)
            self.phi1 = nn.Sequential(self.phi1, max_pool_layer)
            self.phi2 = nn.Sequential(self.phi2, max_pool_layer)

        self.p = Parameter(torch.zeros(1))
        # self.p1 = Parameter(torch.zeros(1))
        # self.p2 = Parameter(torch.zeros(1))

    def forward(self, x, x1, x2, x_in, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        ##abalation
        # if 1:
        #     return torch.cat([x, x1, x2], 1)

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        g_x1 = self.g(x1).view(batch_size, self.inter_channels, -1)
        g_x1 = g_x1.permute(0, 2, 1)

        g_x2 = self.g(x2).view(batch_size, self.inter_channels, -1)
        g_x2 = g_x2.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, -1)

        theta_x1 = self.theta1(x1).view(batch_size, self.inter_channels, -1)
        theta_x1 = theta_x1.permute(0, 2, 1)
        phi_x1 = self.phi1(x1).view(batch_size, self.inter_channels, -1)
        f1 = torch.matmul(theta_x1, phi_x1)
        f_div_C1 = F.softmax(f1, -1)

        theta_x2 = self.theta2(x2).view(batch_size, self.inter_channels, -1)
        theta_x2 = theta_x2.permute(0, 2, 1)
        phi_x2 = self.phi2(x2).view(batch_size, self.inter_channels, -1)
        f2 = torch.matmul(theta_x2, phi_x2)
        f_div_C2 = F.softmax(f2, -1)

        y = torch.matmul(f_div_C, g_x) + torch.matmul(f_div_C1, g_x) + torch.matmul(f_div_C2, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        y1 = torch.matmul(f_div_C, g_x1) + torch.matmul(f_div_C1, g_x1) + torch.matmul(f_div_C2, g_x1)
        y1 = y1.permute(0, 2, 1).contiguous()
        y1 = y1.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y1 = self.W1(y1)
        z1 = W_y1 + x1

        y2 = torch.matmul(f_div_C, g_x2) + torch.matmul(f_div_C1, g_x2) + torch.matmul(f_div_C2, g_x2)
        y2 = y2.permute(0, 2, 1).contiguous()
        y2 = y2.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y2 = self.W2(y2)
        z2 = W_y2 + x2
        # print(1)

        if return_nl_map:
            return z, f_div_C
        return x_in + self.p*self.conv(torch.cat([z, z1, z2],1))




class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)




