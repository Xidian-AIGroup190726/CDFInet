from model.Dfpn import dfpn
# from Tools.GCNK import Graph2dConvolution
import torch.nn as nn
import torch
import math
from model.contourlet_torch import ContourDec
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def conv3x3(in_planes, out_planes, stride=1):
    # 3x3 kernel
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class BasicBlk(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super(BasicBlk, self).__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != self.expansion * out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, self.expansion * out_ch,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_ch)
            )
        self.downsample = downsample

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:  # is not None
            x = self.downsample(x)  # resize the channel
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class Resnet(nn.Module):
    def __init__(self, block, num_blocks, args):
        super(Resnet, self).__init__()
        num_channels = args['num_channels']
        self.in_planes = 32
        self.conv_in = conv3x3(1+num_channels, 32)
        self.BN = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=1)
        self.layer4 = self._make_layer(block, 32, num_blocks[3], stride=1)

        self.conv_out = conv3x3(32, num_channels)
        self.upsample = nn.Upsample(scale_factor=4, mode='nearest')

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, ms, pan):
        hrms = self.upsample(ms)  # [20, 4, 64, 64]
        out = torch.concat((hrms, pan), dim=1)
        out = self.relu(self.BN(self.conv_in(out)))
        out = self.layer1(out)  # torch.Size([20, 64, 16, 16])
        out = self.layer2(out)  # torch.Size([20, 128, 8, 8])
        out = self.layer3(out)  # torch.Size([20, 256, 4, 4])
        out = self.layer4(out)  # torch.Size([20, 512, 2, 2])
        out = torch.sigmoid(self.conv_out(out))
        return out


class Graph2dConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_channels, out_channels, kernel_size, block_num, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros', adj_mask=None
    ):
        super(Graph2dConvolution, self).__init__()
        self.Conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

        self.in_features = in_channels
        self.out_features = out_channels

        self.W = nn.Parameter(torch.randn(in_channels, in_channels))

        self.reset_parameters()
        self.block_num = block_num
        self.adj_mask = adj_mask

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)

    def forward(self, input, index):
        device = input.device
        index = nn.UpsamplingNearest2d(size=(input.shape[2], input.shape[3]))(index.float()).long()

        batch_size = input.shape[0]
        channels = input.shape[1]

        # get one-hot label
        index_ex = torch.zeros(batch_size, self.block_num, input.shape[2], input.shape[3]).to(device)
        index_ex = index_ex.scatter_(1, index, 1)
        block_value_sum = torch.sum(index_ex, dim=(2, 3))

        # computing the regional mean of input
        input_ = input.repeat(self.block_num, 1, 1, 1, 1).permute(1, 0, 2, 3, 4)
        index_ex = index_ex.unsqueeze(2)
        input_means = torch.sum(index_ex * input_, dim=(3, 4)) / (
                    block_value_sum + (block_value_sum == 0).float()).unsqueeze(2)  # * mask.unsqueeze(2)

        # computing the adjance metrix
        input_means_ = input_means.repeat(self.block_num, 1, 1, 1).permute(1, 2, 0, 3)
        input_means_ = (input_means_ - input_means.unsqueeze(1)).permute(0, 2, 1, 3)
        M = (self.W).mm(self.W.T)
        adj = input_means_.reshape(batch_size, -1, channels).matmul(M)
        adj = torch.sum(adj * input_means_.reshape(batch_size, -1, channels), dim=2).view(batch_size, self.block_num,
                                                                                          self.block_num)
        adj = torch.exp(-1 * adj)  # + torch.eye(self.block_num).repeat(batch_size, 1, 1).cuda()
        if self.adj_mask is not None:
            adj = adj * self.adj_mask

        # generating the adj_mean
        adj_means = input_means.repeat(self.block_num, 1, 1, 1).permute(1, 0, 2, 3) * adj.unsqueeze(3)
        adj_means = (1 - torch.eye(self.block_num).reshape(1, self.block_num, self.block_num, 1).to(device)) * adj_means
        adj_means = torch.sum(adj_means, dim=2)  # batch_size，self.block_num, channel_num

        # obtaining the graph update features
        features = torch.sum(index_ex * (input_ + adj_means.unsqueeze(3).unsqueeze(4)), dim=1)

        features = self.Conv2d(features)
        return features

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, ofeat, block_num):
        super(GCN, self).__init__()
        self.convindex = nn.Conv2d(nfeat, block_num, 5, padding=2)
        self.ctindex = ContourDec(0)

        self.gc = Graph2dConvolution(nfeat, ofeat, kernel_size=3, block_num=block_num, padding=1)
        self.bn = nn.BatchNorm2d(ofeat)

    def forward(self, x):
        # print(x.shape)
        index_f = self.convindex(x)
        index_f = self.ctindex(x)[1]
        print(index_f.shape, self.ctindex(x)[0].shape)
        value, index = torch.max(index_f, dim=1, keepdim=True)
        print(index.shape)
        x = self.gc(x, index)
        # print(x.shape)
        x = self.bn(x)
        return x

class GraphClassification(nn.Module):
    def __init__(self, args=1):
        block_num = 5
        channel_num = args['num_channels']
        labelnum = args['Categories']
        super(GraphClassification, self).__init__()
        self.dpfn = dfpn(trans_channel_num=channel_num, resnet_type='resnet18')
        self.gcn = GCN(channel_num, labelnum, block_num=block_num)

    def forward(self, img):
        # features = self.dpfn(img)
        # print(features.shape)
        features = self.gcn(img)
        final_class = torch.mean(features, dim=(2, 3))
        return final_class


class Graphsharpening(nn.Module):
    def __init__(self, args=1):
        block_num = 5
        channel_num = args['num_channels']
        super(Graphsharpening, self).__init__()
        self.dpfn = dfpn(trans_channel_num=channel_num, resnet_type='resnet18')
        self.gcn = GCN(channel_num+1, channel_num, block_num=channel_num+1)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
        self.resnet = Resnet(BasicBlk, [2, 2, 2, 2], args)

        self.convimg = conv3x3(channel_num*2, channel_num)

    def forward(self, m, p):
        hm = self.upsample(m)
        img = torch.concat([hm, p], dim=1)
        # features = self.dpfn(img)
        # print(features.shape)
        out = self.gcn(img)
        # print(img.shape, out.shape)
        out2 = self.resnet(m, p)
        # print(out.shape, out2.shape)
        out = self.convimg(torch.concat((out, out2), dim=1))
        # final_class = torch.mean(features, dim=(2, 3))
        return out


def Net(args):
    return Graphsharpening(args)


if __name__ == '__main__':
    device = 'cuda:1'
    ms = torch.randn([10, 8, 32, 32]).to(device)
    pan = torch.randn([10, 1, 128, 128]).to(device)
    args = {
        'num_channels': 8,
        'Categories': 11
    }
    module = Net(args).to(device)
    result = module(ms, pan)
    print(result.shape)
