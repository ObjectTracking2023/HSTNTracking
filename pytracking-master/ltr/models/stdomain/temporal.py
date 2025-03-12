from torch import nn
from torch.nn import Conv2d, Parameter
from torch.nn import functional as F
import torch


class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            # 在卷积神经网络的卷积层之后总会添加BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch, size):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(size),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class up_sample_three_feature(nn.Module):
    def __init__(self):
        super(up_sample_three_feature, self).__init__()
        self.up1 = up_conv(256,256, (6,6))
        self.up_conv1 = conv_block(512, 256)
        self.up2 = up_conv(256, 256, (9,9))
        self.up_conv2 = conv_block(512, 256)

        self.Conv = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)


    def forward(self, x1, x2, x3):
        r1 = self.up1(x1)
        r1 = self.up_conv1(torch.cat((r1, x2), dim=1))  # 将e4特征图与d5特征图横向拼接
        r2 = self.up2(r1)
        r2 = self.up_conv2(torch.cat((r2, x3), dim=1))
        out = self.Conv(r2)
        return out

class DWConv(nn.Module):
    def __init__(self, in_plane, out_plane):
        super(DWConv, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_plane,out_channels=in_plane,kernel_size=3,stride=1,padding=1,groups=in_plane)
        self.point_conv = nn.Conv2d(in_channels=in_plane,out_channels=out_plane,kernel_size=1,stride=1,padding=0,groups=1)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x

class RMF(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(512, 256, kernel_size=1)
    def forward(self, zf, mf):
        NextRefinedFeature = self.conv(torch.cat([mf, zf], dim=1))
        return NextRefinedFeature

class SFP(nn.Module):
    def __init__(self, in_channel, sizes_test=(9,6,3), sizes_memory=(3,2,1)):
        super().__init__()
        self.stages = []
        self.stages_avg_test = nn.ModuleList(
            [self._make_stage(in_channel, size) for size in sizes_test])
        self.stages_max_test = nn.ModuleList(
            [self._make_stage_max(in_channel, size) for size in sizes_test])

    def _make_stage(self, in_channel, size):
        conv_avg = DWConv(in_channel, in_channel)
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        return nn.Sequential(conv_avg,prior)

    def _make_stage_max(self, in_channel, size):
        conv_max = DWConv(in_channel, in_channel)
        prior = nn.AdaptiveMaxPool2d(output_size=(size, size))
        return nn.Sequential(conv_max, prior)

    def forward(self, feats_test):
        priors_avg_test = [stage(feats_test) for stage in self.stages_avg_test]
        priors_max_test = [stage(feats_test) for stage in self.stages_max_test]
        priors_test = []
        for i in range(0, len(priors_avg_test)):
            sum = priors_avg_test[i] + priors_max_test[i]
            priors_test.append(sum)
        return priors_test

class CTFM(nn.Module):
    def __init__(self):
        super(CTFM, self).__init__()
        self.multihead_attn0 = nn.MultiheadAttention(embed_dim=90, num_heads=9, dropout=0.1)
        self.multihead_attn1 = nn.MultiheadAttention(embed_dim=72, num_heads=9, dropout=0.1)
        self.multihead_attn2 = nn.MultiheadAttention(embed_dim=90, num_heads=9, dropout=0.1)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=81, num_heads=9, dropout=0.1)
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(81)
        self.norm0 = nn.LayerNorm(81)
        self.norm1 = nn.LayerNorm(36)
        self.norm2 = nn.LayerNorm(9)
        self.p = Parameter(torch.zeros(1))
        self.p2 = Parameter(torch.zeros(1))
        self.conv = DWConv(256*3, 256)
        #self.conv2 = DWConv(256*2, 256)
        self.rmf = RMF()
        self.sfp = SFP(256)
        self.vup = up_sample_three_feature()
        self.up3 = up_conv(256, 256, (18, 18))
        self.conv_out = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)

    def forward(self, templates, memorys, searchs):
        temp = []
        for idx, (template, memory, search) in enumerate(zip(templates, memorys, searchs), start=2):

        ##RMF
            Bm, Cm, Hm, Wm = memory.size()
            memory_cropped_feature_out = template + self.p * self.rmf(template, memory)
            ##SFP
            test_list = self.sfp(search)
            memory_list = self.sfp(memory_cropped_feature_out)
            assert len(test_list) == len(memory_list)
            ###MSPT
            v = []
            for i in range(len(test_list)):
                test_list[i] = test_list[i].reshape(Bm, Cm, -1)
                memory_list[2-i] = memory_list[2-i].reshape(Bm, Cm, -1)
                v.append(torch.cat([memory_list[2-i], test_list[i]], dim=2).view(Bm, Cm, -1).permute(1, 0, 2))
            v[0] = self.multihead_attn0(query=v[0].permute(1, 0, 2),key=v[0],value=v[0])[0]
            v[1] = self.multihead_attn1(query=v[1].permute(1, 0, 2),key=v[1],value=v[1])[0]
            v[2] = self.multihead_attn2(query=v[2].permute(1, 0, 2),key=v[2],value=v[2])[0]
            v[0] = v[0]+self.dropout(v[0])
            v[1] = v[1]+self.dropout(v[1])
            v[2] = v[2]+self.dropout(v[2])
            v[0] = v[0][:, :, 0: 3*3]
            v[1] = v[1][:, :, 0: 6*6]
            v[2] = v[2][:, :, 0: 9*9]
            v[0] = self.norm2(v[0]).contiguous().view(Bm, Cm, 3, 3)
            v[1] = self.norm1(v[1]).contiguous().view(Bm, Cm, 6, 6)
            v[2] = self.norm0(v[2]).contiguous().view(Bm, Cm, 9, 9)

            v_up = (self.vup(v[0], v[1], v[2])).reshape(Bm,Cm,-1)

            v_out = self.multihead_attn(query=v_up, key=v_up, value=v_up.reshape(Bm, Cm, -1))[0]
            v_out = self.norm(v_out + self.dropout(v_out)).contiguous().view(Bm,Cm,9,9)
            v_out = self.up3(v_out)
            v_out = self.conv_out(torch.cat((template, v_out), dim = 1))

            updated_template = template + self.p2 * v_out
            temp.append(updated_template)
        return temp
# class CTFM(nn.Module):
#     def __init__(self):
#         super(CTFM, self).__init__()
#         self.multihead_attn0 = nn.MultiheadAttention(embed_dim=65, num_heads=5, dropout=0.1)
#         self.multihead_attn1 = nn.MultiheadAttention(embed_dim=58, num_heads=2, dropout=0.1)
#         self.multihead_attn2 = nn.MultiheadAttention(embed_dim=50, num_heads=5, dropout=0.1)
#         self.multihead_attn = nn.MultiheadAttention(embed_dim=49, num_heads=7, dropout=0.1)
#         self.dropout = nn.Dropout(0.1)
#         self.norm0 = nn.LayerNorm(65)
#         self.norm1 = nn.LayerNorm(58)
#         self.norm2 = nn.LayerNorm(50)
#         self.pa1 = Parameter(torch.zeros(1))
#         self.p = Parameter(torch.zeros(1))
#         self.p2 = Parameter(torch.zeros(1))
#         self.conv = DWConv(256 * 3, 256)
#         # self.conv2 = DWConv(256*2, 256)
#         self.rmf = RMF()
#         self.sfp = SFP(256)
#
#     def forward(self, templates, memorys, searchs):
#         temp = []
#         for idx, (template, memory) in enumerate(zip(templates, memorys), start=2):
#             ##RMF
#             Bm, Cm, Hm, Wm = memory.size()
#             memory_cropped_feature_out = template.reshape(Bm, Cm, -1) + self.p * self.rmf(template, memory).reshape(
#                 Bm, Cm, -1)
#             ##SFP
#             test_list = self.sfp(searchs)
#             ###MSPT
#             test_list[0] = test_list[0].reshape(Bm, Cm, -1)
#             test_list[1] = test_list[1].reshape(Bm, Cm, -1)
#             test_list[2] = test_list[2].reshape(Bm, Cm, -1)
#             v = []
#             v.append(torch.cat([test_list[0], memory_cropped_feature_out], dim=2).view(Bm, Cm, -1).permute(1, 0, 2))
#             v.append(torch.cat([test_list[1], memory_cropped_feature_out], dim=2).view(Bm, Cm, -1).permute(1, 0, 2))
#             v.append(torch.cat([test_list[2], memory_cropped_feature_out], dim=2).view(Bm, Cm, -1).permute(1, 0, 2))
#             v[0] = self.multihead_attn0(query=v[0].permute(1, 0, 2), key=v[0], value=v[0])[0]
#             v[1] = self.multihead_attn1(query=v[1].permute(1, 0, 2), key=v[1], value=v[1])[0]
#             v[2] = self.multihead_attn2(query=v[2].permute(1, 0, 2), key=v[2], value=v[2])[0]
#             v[0] = self.norm0(v[0] + self.dropout(v[0]))
#             v[1] = self.norm1(v[1] + self.dropout(v[1]))
#             v[2] = self.norm2(v[2] + self.dropout(v[2]))
#             v[0] = v[0][:, :, 0: Hm * Wm].contiguous().view(Bm, Cm, Hm, Wm)
#             v[1] = v[1][:, :, 0: Hm * Wm].contiguous().view(Bm, Cm, Hm, Wm)
#             v[2] = v[2][:, :, 0: Hm * Wm].contiguous().view(Bm, Cm, Hm, Wm)
#             v_out = (template + self.p2 * self.conv(torch.cat(v, dim=1))).reshape(Bm, Cm, -1)
#             v_out = self.multihead_attn(query=v_out, key=v_out, value=template.reshape(Bm, Cm, -1))[
#                 0].contiguous().view(Bm, Cm, Hm, Wm)
#             updated_template = self.pa1 * v_out + template
#             temp.append(updated_template)
#         return temp
