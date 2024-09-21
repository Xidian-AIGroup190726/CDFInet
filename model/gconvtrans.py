import torch.nn as nn
import torch
from model.contourlet_torch import ContourDec, lpdec_layer, dfbdec_layer
import numpy as np
import math
# import thop
from torchvision import transforms
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from function.function import to_tensor
from PIL import Image
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from model.gcnn import GCN, conv3x3, BasicBlk
from model.swin_ct import (BasicLayer, WindowAttention, ContourletAttention,
                           PatchMerging, Mlp, window_partition, window_reverse)
from model.gctrans import PatchExpand, PatchReduce
import time


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
        # import matplotlib.pyplot as plt
        # import seaborn as sns
        # # plt.imshow(adj_means[0].cpu().detach().numpy(), cmap='gray')
        # sns.heatmap(adj_means[0].cpu().detach().numpy(), cmap='gray', annot=True)
        # plt.tick_params(axis='both', which='both', bottom=True, top=True, left=True, right=True)  # 隐藏刻度线
        # plt.xticks([])  # 隐藏x轴刻度标签
        # plt.yticks([])
        # plt.savefig("adj.jpg")
        # plt.clf()
        # plt.colorbar()  # 显示颜色条
        # plt.title('64x64 Matrix Visualization')
        # plt.savefig("adj.jpg")
        # plt.show()


        # obtaining the graph update features
        features = torch.sum(index_ex * (input_ + adj_means.unsqueeze(3).unsqueeze(4)), dim=1)

        features = self.Conv2d(features)
        return features

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


def equalize_histogram(band):
    hist, bins = np.histogram(band.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    return cdf[band]


def xianhua(img, name, equalize=1):
    img = img.cpu().detach().numpy()
    img = to_tensor(img)
    if img.shape[0] == 4 :
        band_data = img[(2, 1, 0), :, :]
        scaled_data = []
        for i, band in enumerate(band_data):
            band_min, band_max = band.min(), band.max()
            scaled_band = ((band - band_min) / (band_max - band_min) * 255).astype(np.uint8)
            if equalize:
                scaled_band = equalize_histogram(scaled_band)
            # scaled_band = adjust_contrast(scaled_band, 0.95)
            # scaled_band = adjust_exposure(scaled_band, 0.95)
            scaled_data.append(scaled_band)

        processed_array = np.dstack(scaled_data)
        # processed_array = adjust_brightness_hsv(processed_array, 0.85)
    elif img.shape[0] == 8:
        band_data = img[(4, 2, 1), :, :]
        # band_data = img[(2, 1, 0), :, :]
        scaled_data = []
        for i, band in enumerate(band_data):
            band_min, band_max = band.min(), band.max()
            scaled_band = ((band - band_min) / (band_max - band_min) * 255).astype(np.uint8)
            if equalize:
                scaled_band = equalize_histogram(scaled_band)
            # scaled_band = adjust_contrast(scaled_band, 0.95)
            # scaled_band = adjust_exposure(scaled_band, 0.95)
            scaled_data.append(scaled_band)
        processed_array = np.dstack(scaled_data)
    elif img.shape[0] == 1:
        band = img[0]
        band_min, band_max = band.min(), band.max()
        processed_band = ((band - band_min) / (band_max - band_min) * 255).astype(np.uint8)
        if equalize:
            processed_band = equalize_histogram(processed_band)
        processed_array = processed_band

    else:
        raise ValueError("Unsupported image type. Please use 'multispectral' or 'pan'.")

    result = Image.fromarray(processed_array, 'RGB' if img.shape[0] == 4 or img.shape[0] == 8 else 'L')
    # result.show()
    result.save(name)
    

def visualize_channels(tensor_in, num_channels=8, cols=4, name=''):
    def mapping(x):
        x_min = x.min()
        x_max = x.max()

        # 将x的值映射到0-255
        x_normalized = (x - x_min) / (x_max - x_min)  # 归一化到[0, 1]
        # x_mapped = (x_normalized * 255)
        return x_normalized

    """
    可视化指定数量的通道。
    :param tensor: BCHW 形状的张量。
    :param num_channels: 要展示的通道数量。
    :param cols: 每行显示的图像数量。
    """
    from matplotlib.colors import LinearSegmentedColormap
    # colors_list = [(0, '#0000FF'), # 蓝色
    #             # (0.25, '#00FFFF'), # 青色
    #             (0.5, '#00FF00'),  # 绿色
    #             # (0.15, '#BFFF00'), # 黄绿色
    #             # (0.75, '#FFFF00'), # 黄色
    #             (1, '#FF0000')] # 红色
    colors = [(0, '#0000FF'),
              (0.2, '#0055FF'),
              (0.4, '#00FFFF'),
              (0.6, '#00FFAA'),
              (0.8, '#AAFF00'),
              (1, '#FFFF00')]
    # colors_list = [(0, '#0000FF'),  # 蓝色
    #                (1, '#FF0000')]  # 红色
    # cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
    color = "seismic" # "seismic"
    import matplotlib.pyplot as plt
    tensor = tensor_in[0]  # 选择批次中的第一个样本
    channels = tensor.shape[0]  # 获取通道数
    # 如果通道数为1，仅展示这一个通道
    if channels == 1:
        plt.imshow(tensor[0].cpu().detach().numpy(), cmap=color)
        plt.axis('off')
        plt.title('Single_'+name)
        plt.savefig('Single_'+name)
        plt.show()
        return

    rows = num_channels // cols + int(num_channels % cols > 0)

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
    axes = axes.flatten()

    for i in range(num_channels):
        ax = axes[i]
        ax.imshow(tensor[i].cpu().detach().numpy(), cmap=color)
        ax.axis('off')
        ax.set_title(f'Channel {i + 1}-'+name)

    for i in range(num_channels, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(f'Channel {i + 1}-'+name)
    # plt.colorbar()
    plt.show()
    plt.close()


class Resnet(nn.Module):
    def __init__(self, block, num_blocks, args):
        super(Resnet, self).__init__()
        num_channels = args['in_channels']
        in_chan = args['num_channels']
        self.in_planes = in_chan
        self.conv_in = conv3x3(1+num_channels, in_chan)
        self.BN = nn.BatchNorm2d(in_chan)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, in_chan, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, in_chan*2, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(block, in_chan*2, num_blocks[2], stride=1)
        self.layer4 = self._make_layer(block, in_chan*2, num_blocks[3], stride=1)

        self.conv_out = conv3x3(in_chan*2, num_channels)
        self.upsample = nn.Upsample(scale_factor=4, mode='nearest')

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def encoder(self, out):
        out = self.layer1(out)  # torch.Size([20, 64, 16, 16])
        out = self.layer2(out)  # + graph1  # torch.Size([20, 128, 8, 8])
        return out

    def decoder(self, out):
        out = self.layer3(out)  # + graph2  # torch.Size([20, 256, 4, 4])
        out = self.layer4(out)  # torch.Size([20, 512, 2, 2])
        out = torch.sigmoid(self.conv_out(out))
        return out

    def forward(self, hrms, pan):
        # hrms = self.upsample(ms)  # [20, 4, 64, 64]
        out = torch.concat((hrms, pan), dim=1)
        # out = hrms + pan
        # out = self.relu(self.BN(out))
        out = self.layer1(out)  # torch.Size([20, 64, 16, 16])
        out = self.layer2(out)  # + graph1  # torch.Size([20, 128, 8, 8])
        # out = self.decoder(out)
        return out


class GCTN(nn.Module):
    def __init__(self, in_chan, chan_num, bl_num, num_chan):
        super(GCTN, self).__init__()
        self.gc = Graph2dConvolution(in_chan, chan_num, kernel_size=3, block_num=bl_num, padding=1)
        self.convin = nn.Conv2d(num_chan, num_chan, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(chan_num)

    def forward(self, x, index_f):
        # index_f = lpdec_layer(x)[1]
        value, index = torch.max(self.convin(index_f), dim=1, keepdim=True)
        # print(value, index)
        x = self.gc(x, index)
        x = self.bn(x)
        return x


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=128, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class EncodeBlock(nn.Module):
    def __init__(self, dim, input_resolution, depth, attn_mode, exc_mode, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 pretrained_window_size=0, device='cuda'):
        super().__init__()
        self.device = device
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            Dual_T_Block(dim=dim, input_resolution=input_resolution,
                         num_heads=num_heads, window_size=window_size,
                         shift_size=0 if (i % 2 == 0) else window_size // 2,
                         attn_mode=attn_mode,
                         exc_mode=exc_mode,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias,
                         drop=drop, attn_drop=attn_drop,
                         drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                         norm_layer=norm_layer,
                         pretrained_window_size=pretrained_window_size,
                         device=self.device
                         )
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, m, p, m_coefs=None, p_coefs=None):
        for blk in self.blocks:
            if self.use_checkpoint:
                m = checkpoint.checkpoint(blk, m)
                p = checkpoint.checkpoint(blk, p)
            else:
                m, p = blk(m, p, m_coefs, p_coefs)
        if self.downsample is not None:
            m = self.downsample(m)
            p = self.downsample(p)
        return m, p

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


class DecodeBlock(nn.Module):
    def __init__(self, dim, input_resolution, depth, attn_mode, exc_mode, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 pretrained_window_size=0, device='cuda'):
        super().__init__()
        self.device = device
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            Dual_T_Block(dim=dim, input_resolution=input_resolution,
                         num_heads=num_heads, window_size=window_size,
                         shift_size=0 if (i % 2 == 0) else window_size // 2,
                         attn_mode=attn_mode,
                         exc_mode=exc_mode,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias,
                         drop=drop, attn_drop=attn_drop,
                         drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                         norm_layer=norm_layer,
                         pretrained_window_size=pretrained_window_size,
                         device=self.device
                         )
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, m, p, m_coefs=None, p_coefs=None):
        for blk in self.blocks:
            if self.use_checkpoint:
                m = checkpoint.checkpoint(blk, m)
                p = checkpoint.checkpoint(blk, p)
            else:
                m, p = blk(m, p, m_coefs, p_coefs)
        if self.downsample is not None:
            m = self.downsample(m)
            p = self.downsample(p)
        return m, p

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


class Dual_T_Block(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, attn_mode, exc_mode, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0, device='cuda'):
        super().__init__()
        self.device = device
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = window_attn(
            dim, attn_mode=attn_mode, exc_mode=exc_mode, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size),
            device=self.device
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_m = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp_p = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, m, p, m_coefs=None, p_coefs=None):
        # H, W = self.input_resolution
        B, L, C = m.shape
        H, W = int(math.sqrt(L)), int(math.sqrt(L))
        # assert L == H * W, "input feature has wrong size"
        # print("before swin", x.shape, y[0].shape)
        m_shortcut, p_shortcut = m, p
        # print('dual', m.shape, m_band[0])
        m, p = m.view(B, H, W, C), p.view(B, H, W, C)
        # print(m.shape)
        if m_coefs:
            m_band, p_band = m_coefs, p_coefs
            m_band[0] = m_band[0].view(B, H // 2, W // 2, C)
            p_band[0] = p_band[0].view(B, H // 2, W // 2, C)
            if len(m_band) == 2:
                m_band[1] = m_band[1].view(B, H, W, -1)
                p_band[1] = p_band[1].view(B, H, W, -1)

        # cyclic shift
        if self.shift_size > 0:
            shifted_m = torch.roll(m, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_p = torch.roll(p, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            if m_coefs:
                m_band[0] = torch.roll(m_band[0], shifts=(-self.shift_size // 2, -self.shift_size // 2), dims=(1, 2))
                p_band[0] = torch.roll(p_band[0], shifts=(-self.shift_size // 2, -self.shift_size // 2), dims=(1, 2))
                if len(m_band) == 2:
                    m_band[1] = torch.roll(m_band[1], shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                    p_band[1] = torch.roll(p_band[1], shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_m, shifted_p = m, p

        # partition windows
        m_windows = window_partition(shifted_m, self.window_size)  # nW*B, window_size, window_size, C
        m_windows = m_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        p_windows = window_partition(shifted_p, self.window_size)  # nW*B, window_size, window_size, C
        p_windows = p_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        if m_coefs:
            m_band[0] = window_partition(m_band[0], self.window_size // 2)
            m_band[0] = m_band[0].view(-1, self.window_size // 2 * self.window_size // 2, C)
            p_band[0] = window_partition(p_band[0], self.window_size // 2)
            p_band[0] = p_band[0].view(-1, self.window_size // 2 * self.window_size // 2, C)
            if len(m_band) == 2:
                m_band[1] = window_partition(m_band[1], self.window_size)
                m_band[1] = m_band[1].view(m_band[0].shape[0], self.window_size * self.window_size, -1)
                p_band[1] = window_partition(p_band[1], self.window_size)
                p_band[1] = p_band[1].view(p_band[0].shape[0], self.window_size * self.window_size, -1)
        # W-MSA/SW-MSA
        # print("before attn", m_windows.shape, m_band[0].shape)
        m_attn_windows, p_attn_windows = self.attn(m_windows, p_windows, m_band, p_band, mask=self.attn_mask)
        # visualize_channels(m.transpose(1,2).view(B, -1, H, W), 32, 4, f'{i}_m_decode.png')
        # m_attn_windows, p_attn_windows = self.attn(m_windows, p_windows, m_coefs, p_coefs, mask=self.attn_mask)

        # merge windows
        m_attn_windows = m_attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_m = window_reverse(m_attn_windows, self.window_size, H, W)  # B H' W' C
        p_attn_windows = p_attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_p = window_reverse(p_attn_windows, self.window_size, H, W)  # B H' W' C
        # visualize_channels(shifted_m.permute(0, 3, 1, 2), 32, 4, str(len(m_band))+str(time.time())+'m_map.png')
        # visualize_channels(shifted_p.permute(0, 3, 1, 2), 32, 4, str(len(m_band))+str(time.time())+'p_map.png')

        # reverse cyclic shift
        if self.shift_size > 0:
            m = torch.roll(shifted_m, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            p = torch.roll(shifted_p, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            m, p = shifted_m, shifted_p

        m, p = m.view(B, H * W, C), p.view(B, H * W, C)
        m = m_shortcut + self.drop_path(self.norm1(m))
        p = p_shortcut + self.drop_path(self.norm1(p))

        m = m + self.drop_path(self.norm2(self.mlp_m(m)))
        p = p + self.drop_path(self.norm2(self.mlp_p(p)))
        return m, p

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class window_attn(nn.Module):
    def __init__(self, dim, window_size, num_heads, attn_mode, exc_mode, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=[0, 0], device='cuda'):
        super().__init__()
        self.device = device
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads
        self.attn_mode = attn_mode
        self.exc_mode = exc_mode

        self.m_logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)
        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(512, num_heads, bias=False))

        self.p_logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)
        # mlp to generate continuous relative position bias

        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h,
                            relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.m_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.p_qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.m_q_bias, self.m_v_bias = nn.Parameter(torch.zeros(dim)), nn.Parameter(torch.zeros(dim))
            self.p_q_bias, self.p_v_bias = nn.Parameter(torch.zeros(dim)), nn.Parameter(torch.zeros(dim))
        else:
             self.m_q_bias, self.m_v_bias = None, None
             self.p_q_bias, self.p_v_bias = None, None
        self.m_attn_drop = nn.Dropout(attn_drop)
        self.m_proj = nn.Linear(dim, dim)
        self.m_proj_drop = nn.Dropout(proj_drop)
        self.m_softmax = nn.Softmax(dim=-1)
        self.m_anchor = nn.Linear(dim, dim, bias=False)

        self.p_attn_drop = nn.Dropout(attn_drop)
        self.p_proj = nn.Linear(dim, dim)
        self.p_proj_drop = nn.Dropout(proj_drop)
        self.p_softmax = nn.Softmax(dim=-1)
        self.p_anchor = nn.Linear(dim, dim, bias=False)

    def forward(self, m, p, m_coefs=None, p_coefs=None, mask=None):
        # print("bnc", m.shape)
        B_, N, C = m.shape

        # Normal qkv definition
        m_qkv_bias, p_qkv_bias = None, None
        if self.m_q_bias is not None or self.p_q_bias is not None:
            m_qkv_bias = torch.cat((self.m_q_bias, torch.zeros_like(self.m_v_bias,
                                                                    requires_grad=False), self.m_v_bias))
            p_qkv_bias = torch.cat((self.p_q_bias, torch.zeros_like(self.p_v_bias,
                                                                    requires_grad=False), self.p_v_bias))
        m_qkv = F.linear(input=m, weight=self.m_qkv.weight, bias=m_qkv_bias)
        m_qkv = m_qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        m_q, m_k, m_v = m_qkv[0], m_qkv[1], m_qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        p_qkv = F.linear(input=p, weight=self.p_qkv.weight, bias=p_qkv_bias)
        p_qkv = p_qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        p_q, p_k, p_v = p_qkv[0], p_qkv[1], p_qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # direction definition
        if self.attn_mode:
            m_s = m_coefs[1].reshape(B_, N, self.num_heads, -1).permute(0, 2, 1, 3)
            p_s = p_coefs[1].reshape(B_, N, self.num_heads, -1).permute(0, 2, 1, 3)
        # attention calculation
        if m_coefs is not None or p_coefs is not None:
            # anchor definition
            m_l = m_coefs[0]
            p_l = p_coefs[0]
            # print("lowsubband:", m_l.shape, p_l.shape)
            m_anchor = self.m_anchor(p_l if self.exc_mode else m_l)
            m_a = m_anchor.reshape(B_, N // 4, self.num_heads, -1).permute(0, 2, 1, 3)
            p_anchor = self.p_anchor(m_l if self.exc_mode else p_l)
            p_a = p_anchor.reshape(B_, N // 4, self.num_heads, -1).permute(0, 2, 1, 3)
            # print("anchor:", m_a.shape, p_a.shape)
            # print(m_q.shape)

            m_attn1 = (F.normalize(m_q, dim=-1) @ F.normalize(m_a, dim=-1).transpose(-2, -1))
            m_attn2 = (F.normalize(m_k, dim=-1) @ F.normalize(m_a, dim=-1).transpose(-2, -1))
            m_attn = (F.normalize(m_attn1, dim=-1) @ F.normalize(m_attn2, dim=-1).transpose(-2, -1))

            p_attn1 = (F.normalize(p_q, dim=-1) @ F.normalize(p_a, dim=-1).transpose(-2, -1))
            p_attn2 = (F.normalize(p_k, dim=-1) @ F.normalize(p_a, dim=-1).transpose(-2, -1))
            p_attn = (F.normalize(p_attn1, dim=-1) @ F.normalize(p_attn2, dim=-1).transpose(-2, -1))
            p_logit_scale = torch.clamp(self.p_logit_scale,
                                        max=torch.log(torch.tensor(1. / 0.01).to(self.device))).exp()
            p_attn = p_attn * p_logit_scale
        else:
            # original attention
            m_attn = (F.normalize(m_q, dim=-1) @ F.normalize(m_k, dim=-1).transpose(-2, -1))
            m_logit_scale = torch.clamp(self.m_logit_scale,
                                        max=torch.log(torch.tensor(1. / 0.01).to(self.device))).exp()
            m_attn = m_attn * m_logit_scale

            p_attn = (F.normalize(p_q, dim=-1) @ F.normalize(p_k, dim=-1).transpose(-2, -1))

        m_logit_scale, p_logit_scale = (torch.clamp(self.m_logit_scale,
                                        max=torch.log(torch.tensor(1. / 0.01).to(self.device))).exp(),
                                         torch.clamp(self.p_logit_scale,
                                        max=torch.log(torch.tensor(1. / 0.01).to(self.device))).exp())
        p_attn, m_attn = p_attn * p_logit_scale, m_attn * m_logit_scale

        # relative position encode
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1],
            -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        m_attn = m_attn + relative_position_bias.unsqueeze(0)
        p_attn = p_attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            m_attn = m_attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            m_attn = self.m_softmax(m_attn.view(-1, self.num_heads, N, N))
            p_attn = p_attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            p_attn = self.p_softmax(p_attn.view(-1, self.num_heads, N, N))
        else:
            m_attn = self.m_softmax(m_attn)
            p_attn = self.p_softmax(p_attn)
            m_v = torch.mul(m_v, m_s) if self.attn_mode else m_v
            p_v = torch.mul(p_v, p_s) if self.attn_mode else p_v
            # visualize_channels(m_v.permute(0, 3, 1, 2), 32, 4, str(time.time()) + 'm_v.png')
            # visualize_channels(p_v.permute(0, 3, 1, 2), 32, 4, str(time.time()) + 'p_v.png')


        if self.exc_mode:
            temp = m_v
            p_v, m_v = temp, p_v

        m = ((m_attn @ m_v).transpose(1, 2).reshape(B_, N, C))
        p = ((p_attn @ p_v).transpose(1, 2).reshape(B_, N, C))
        m, p = self.m_proj(m), self.p_proj(p)
        m, p = self.m_proj_drop(m), self.p_proj_drop(p)
        return m, p


class Graphsharpening(nn.Module):
    def __init__(self, patch_size=4, in_chans=4, num_classes=0, embed_dim=96, depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=False, patch_norm=True, use_checkpoint=False,
                 pretrained_window_sizes=[0, 0, 0, 0], n_levs=[4, 3], attn_mode=[0, 1, 1, 1],
                 exc_mode=[0, 1, 1, 1], device='cuda', **kwargs):
        super(Graphsharpening, self).__init__()
        self.upsample = nn.Upsample(scale_factor=4, mode='nearest')

        # self.m_in = nn.Conv2d(in_chans, int(embed_dim//2), patch_size, patch_size)
        # self.p_in = nn.Conv2d(1, int(embed_dim//2), patch_size, patch_size)
        self.num_chan = embed_dim
        self.m_in = nn.Conv2d(in_chans, int(self.num_chan//2), 1, 1)
        self.p_in = nn.Conv2d(1, int(self.num_chan//2), 1, 1)
        self.BN = nn.BatchNorm2d(embed_dim)
        self.relu = nn.ReLU(inplace=True)
        # self.resnet = Resnet(BasicBlk, [2, 2, 2, 2], args)
        self.convimg = conv3x3(embed_dim, in_chans)

        self.patch_size = patch_size
        self.num_layers = len(depths) // 2
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.device = device

        self.graph_init(in_chans)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
        self.n_levs = n_levs
        self.n_tims = len(n_levs)
        self.attn_mode = attn_mode
        self.in_chans_list = []
        self.m_patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans,
                                        embed_dim=embed_dim,
                                        norm_layer=norm_layer if self.patch_norm else None)
        self.p_patch_embed = PatchEmbed(patch_size=patch_size, in_chans=1, embed_dim=embed_dim,
                                        norm_layer=norm_layer if self.patch_norm else None)
        self.m_linear = nn.Linear(embed_dim, embed_dim, True)
        self.p_linear = nn.Linear(embed_dim, embed_dim, True)
        num_patches = self.m_patch_embed.num_patches
        patches_resolution = self.m_patch_embed.patches_resolution

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.conv_mg = nn.Conv2d(int(self.num_chan//2), embed_dim, 1, 1)
        self.conv_pg = nn.Conv2d(int(self.num_chan//2), embed_dim, 1, 1)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.Encoder = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # print(i_layer, int(embed_dim * 2 ** i_layer))
            layer = EncodeBlock(dim=embed_dim,
                                #dim=int(embed_dim * 2 ** i_layer),
                                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                  patches_resolution[1] // (2 ** i_layer)),
                                # input_resolution=(patches_resolution[0], patches_resolution[1]),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                attn_mode=attn_mode[i_layer],
                                exc_mode=exc_mode[i_layer],
                                window_size=window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias,
                                drop=drop_rate, attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=None,  # PatchMerging,PatchExpand
                                use_checkpoint=use_checkpoint,
                                pretrained_window_size=pretrained_window_sizes[i_layer],
                                device=device)
            self.Encoder.append(layer)

        self.Decoder = nn.ModuleList()
        for i_layer in range(self.num_layers):
            plus_layer = i_layer + self.num_layers
            layer = DecodeBlock(dim=int(embed_dim),
                                #dim=int(embed_dim * 2 ** (self.num_layers - i_layer)),
                                input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - i_layer)),
                                                  patches_resolution[1] // (2 ** (self.num_layers - i_layer))),
                                # input_resolution=(patches_resolution[0], patches_resolution[1]),
                                depth=depths[plus_layer],
                                num_heads=num_heads[plus_layer],
                                attn_mode=attn_mode[plus_layer],
                                exc_mode=exc_mode[plus_layer],
                                window_size=window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias,
                                drop=drop_rate, attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[self.num_layers:plus_layer]):sum(depths[self.num_layers:plus_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=None,  # PatchSplitting, PatchReduce
                                use_checkpoint=use_checkpoint,
                                pretrained_window_size=pretrained_window_sizes[i_layer],
                                device=device)
            self.Decoder.append(layer)

        self.norm = norm_layer(self.embed_dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.convblock = nn.ModuleList()
        if self.patch_size == 1:
            layer = nn.Conv2d(self.embed_dim, in_chans, 3, 1, 1)
            self.convblock.append(layer)
        else:
            for i_layer in range(int(math.log2(self.patch_size))):
                layer = nn.ConvTranspose2d(self.embed_dim // (2 ** (i_layer)),
                                           in_chans if i_layer == int(
                                               math.log2(self.patch_size)) - 1 else self.embed_dim // (
                                                   2 ** (i_layer + 1)), 4, 2, 1)
                self.convblock.append(layer)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.conv4 = nn.Conv2d(1, in_chans, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_chans, in_chans, 3, 1, 1)
        self.conv5 = nn.Conv2d(in_chans*2, in_chans,  1, 1)

        self.resnet = Resnet(BasicBlk, [1, 1, 1, 1], {"in_channels": in_chans, "num_channels":self.num_chan})

    def forward_feature2img(self, x, y):
        for layer in self.convblock:
            x = layer(x)
            # print(x.shape)
        # x = self.pixel_shuffle(x)
        # print(x.shape)
        # x = self.conv4(z) + self.conv3(y)
        # x = self.conv5(torch.cat([x, y], dim=1))
        # self.vs_c(y[0][0], 'y.jpg')
        # visualize_channels(y, 4, 2, f'y.png')
        x = y - x
        # visualize_channels(x, 4, 2, f'trans_out2.png')
        return x

    def graph_init(self, in_channel):
        self.m_bk1_1 = GCTN(in_channel, in_channel, in_channel, in_channel)
        self.p_bk1_1 = GCTN(1, in_channel, in_channel, 1)

        self.m_bk1_2 = GCTN(in_channel, in_channel, in_channel, in_channel)
        self.p_bk1_2 = GCTN(in_channel, in_channel, in_channel, in_channel)

        self.m_bk2_1 = GCTN(in_channel, int(self.num_chan // 2), in_channel*4, in_channel*4)
        self.p_bk2_1 = GCTN(in_channel, int(self.num_chan // 2), in_channel*4, 4)

        # self.m_bk2_2 = GCTN(int(self.num_chan // 2), int(self.num_chan // 2), in_channel * 4)
        # self.p_bk2_2 = GCTN(int(self.num_chan // 2), self.num_chan, in_channel * 4)
        # self.convdownsample = nn.Conv2d(in_channel*8, in_channel*8, 3, 2, 1)
        # self.bn = nn.BatchNorm2d(self.num_chan)
        # self.m_gcn2 = GCTN_lp(channel_num, channel_num*4, block_num=channel_num*4)
        self.gcn1 = GCTN(self.num_chan, self.num_chan, in_channel*4, in_channel*4)
        # self.gcn1 = GCTN(self.embed_dim, self.embed_dim, in_channel * 4)

        # self.m_bk3 = GCTN(in_channel*4, in_channel*4, in_channel)
        # self.p_bk3 = GCTN(in_channel*4, in_channel*4, in_channel)
        # self.m_bk4 = GCTN(in_channel*4, in_channel*4, in_channel*4)
        # self.p_bk4 = GCTN(in_channel*4, in_channel*4, in_channel*4)
        # self.gcn2 = GCTN(in_channel*8, self.embed_dim, in_channel*4)

    def dfb(self, x):
        xhi = dfbdec_layer(x, 2)
        for i in range(len(xhi)):
            k = xhi[i].shape
            if k[2] != k[3]:
                # Get maximum dimension (height or width)
                max_dim = int(np.max((k[2], k[3])))
                # Resize the channels
                trans = transforms.Compose([transforms.Resize((max_dim, max_dim))])
                xhi[i] = trans(xhi[i])
        xhi = torch.cat(xhi, dim=1)
        return xhi

    def ct(self, img):
        B, C, H, W = img.shape
        coefs = []
        l, h = lpdec_layer(img)
        dh = self.dfb(h)
        # print(dh.shape)
        # visualize_channels(dh, 4, 2, 's.png')
        # coefs.append([h, dh])
        # print(l.shape, dh.shape)
        # proj_l = nn.Conv2d(C, int(self.embed_dim * (2 ** i)), 1,1).to(img.device)
        # proj_s = nn.Conv2d(C * 4, self.num_heads[i], 1, 1).to(img.device)
        proj_l = nn.Conv2d(C, self.embed_dim, self.patch_size, self.patch_size).to(img.device)
        proj_s = nn.Conv2d(C * 4, self.embed_dim, 1, 1).to(img.device)

        l_e = proj_l(l).flatten(2).transpose(1, 2)
        s_e = proj_s(dh).flatten(2).transpose(1, 2)
        s_h = nn.UpsamplingNearest2d(size=(H, W))(proj_s(dh))
        # print("ct", l_e.shape, s_e.shape)
        coefs.append([h, dh, l_e, s_e, s_h])
        # img = l
        return coefs
            # # proj_s = nn.Conv2d(C * (2**self.n_levs[i]), self.embed_dim*(2**(i)),
            # #                    int(self.patch_size/2), int(self.patch_size/2)).to(self.device)
            # l_1, l_2 = proj_l(l_1).flatten(2).transpose(1, 2), proj_l(l_2).flatten(2).transpose(1, 2)
            # l_1, l_2 = proj_l(l_1).flatten(2).transpose(1, 2), proj_l(l_2).flatten(2).transpose(1, 2)

        # self.m_l1, self.m_h1 = lpdec_layer(m)
        # self.p_l1, self.p_h1 = lpdec_layer(p)
        # self.m_dh1 = self.dfb(self.m_h1)
        # self.p_dh1 = self.dfb(self.p_h1)
        #
        # self.m_l2, self.m_h2 = lpdec_layer(self.m_l1)
        # self.p_l2, self.p_h2 = lpdec_layer(self.p_l1)
        # self.m_dh2 = self.dfb(self.m_h2)
        # self.p_dh2 = self.dfb(self.p_h2)

    def graph(self, m, p):
        _, C, _, _ = m.shape
        #graph 1
        #visualize_channels(m, 8, 4, 'm')
        #visualize_channels(p, 1, 4, 'p')
        m_out = self.m_bk1_1(m, self.m_coefs[0][0])
        p_out = self.p_bk1_1(p, self.p_coefs[0][0])

        #visualize_channels(self.p_coefs[0][1] ** 2, 4, 2, 'ps')
        #visualize_channels(self.m_coefs[0][1] ** 2, 4, 2, 'ms')
        # m_out_c = self.m_bk1_2(m_out, self.p_coefs[0][0])
        # p_out_c = self.p_bk1_2(p_out, self.m_coefs[0][0])
        #visualize_channels(m_out, 8, 4, 'm_out1')
        #visualize_channels(p_out, 8, 4, 'p_out1')
        m_out = self.m_bk2_1(m_out, self.m_coefs[0][1])
        p_out = self.p_bk2_1(p_out, self.p_coefs[0][1])

        #visualize_channels(m_out, 8, 4, 'm_out2')
        #visualize_channels(p_out, 8, 4, 'p_out2')
        # print(m_out.shape, p_out.shape)
        # graph cross
        p_dh1_expanded = self.p_coefs[0][1].repeat_interleave(C, dim=1)
        cross_index1 = self.m_coefs[0][1] * p_dh1_expanded
        # out = self.bn(self.convdownsample(torch.concat([m_out, p_out], dim=1)))
        out = torch.concat([m_out, p_out], dim=1)
        out = self.gcn1(out, cross_index1)
        m_cout, p_cout = torch.chunk(out, 2, 1)
        #visualize_channels(m_cout, 8, 4, 'm_cout')
        #visualize_channels(p_cout, 8, 4, 'p_cout')
        # return torch.cat([m_out - m_cout, p_out - p_cout], dim=1)
        #visualize_channels(m_out - m_cout, 8, 4, 'm_out - m_cout')
        #visualize_channels(p_out - p_cout, 8, 4, 'p_out - p_cout')
        return m_out - m_cout, p_out - p_cout
        # visualize_channels(out1, 16, 4, 'cross_index')

        # cross module
        # m_out = self.m_bk3(m_out, self.m_coefs[1][0])
        # p_out = self.p_bk3(p_out, self.p_coefs[1][0])
        # m_out = self.m_bk4(m_out, self.m_coefs[1][1])
        # p_out = self.p_bk4(p_out, self.p_coefs[1][1])
        #
        # p_dh2_expanded = self.p_coefs[1][1].repeat_interleave(C, dim=1)
        # cross_index2 = self.m_coefs[1][1] * p_dh2_expanded
        # # out = self.bn(self.convdownsample(torch.concat([m_out, p_out], dim=1)))
        # out2 = self.gcn2(out, cross_index2)
        # return out1 #out1.flatten(2).transpose(1, 2), out2.flatten(2).transpose(1, 2)
        
    def vs_c(self, out, adr):
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.colors import LinearSegmentedColormap
        colors = [(0, '#0000FF'),
                    (0.25, '#00FFFF'),
                    (0.5, '#00FF00'),  # 绿色
                    (0.75, '#BFFF00'),
                    (1, '#FFFF00')]  # 红色
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
        from matplotlib.colors import LinearSegmentedColormap
        sns.heatmap(out.cpu().detach().numpy(), cmap=cmap, annot=False)  # cmap可以选择其他颜色映射，annot为True会显示数值标签
        plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)  # 隐藏刻度线
        plt.xticks([])  # 隐藏x轴刻度标签
        plt.yticks([])
        plt.savefig(adr)
        plt.clf()
    

    def forward(self, lms, pan):
        hm = self.upsample(lms)  # (20, 4, 128, 128)
        B, C, H, W = hm.shape
        # contourlet decompose
        self.m_coefs, self.p_coefs = self.ct(hm), self.ct(pan)

        graph_out = self.graph(hm, pan)
        m1, p1 = self.m_in(hm), self.p_in(pan)
        # visualize_channels(m1, 16, 4, "after_m1")
        # visualize_channels(p1, 16, 4, "after_p1")
        # visualize_channels(graph_out[1], 16, 4, "graph_m1")
        # visualize_channels(graph_out[0], 16, 4, "graph_p1")
        # visualize_channels(m1 + graph_out[1], 16, 4, "m1+graph_m1")
        # visualize_channels(p1 + graph_out[0], 16, 4, "p1+graph_p1")
        out = self.resnet(m1 + graph_out[1], p1 + graph_out[0])
        # out = self.resnet(m1, p1)
        # visualize_channels(out, 16, 4, "out")

        # visualize_channels(hm, 4, 2, f'hrms.png')
        # visualize_channels(pan, 1, 1, 'pan.png')

        #visualize_channels(hm - self.m_coefs[0][0], 4, 2, f'm_lo.png')
        #visualize_channels(pan - self.p_coefs[0][0], 1, 1, f'p_lo.png')

        m_embed = self.m_patch_embed(hm - self.m_coefs[0][0])  # (20, 1024, 64)
        p_embed = self.p_patch_embed(pan - self.p_coefs[0][0])  # (20, 1024, 64)

        #visualize_channels(m_embed.transpose(1,2).view(B, -1, H, W), 32, 4, f'm_embed.png')
        #visualize_channels(p_embed.transpose(1,2).view(B, -1, H, W), 32, 4, f'p_embed.png')

        m = self.pos_drop(self.m_linear(m_embed))
        p = self.pos_drop(self.p_linear(p_embed))
        # visualize_channels(m.transpose(1,2).view(B, -1, H, W), 32, 4, f'm_encode.png')
        # visualize_channels(p.transpose(1,2).view(B, -1, H, W), 32, 4, f'p_encode.png')
        m_list, p_list = [], []
        for i in range(self.num_layers):
            m_list.append(m)
            p_list.append(p)
            # m, p = self.Encoder[i](m, p, [self.m_coefs[i][2]], [self.p_coefs[i][2]]) # if self.attn_mode[i] else self.Encoder[i](m, p)
            m, p = self.Encoder[i](m, p, [self.m_coefs[0][2]], [self.p_coefs[0][2]])
            # visualize_channels(m.transpose(1, 2).view(B, -1, H, W), 32, 4, f'{i}_m_encode.png')
            # visualize_channels(p.transpose(1, 2).view(B, -1, H, W), 32, 4, f'{i}_p_encode.png')
        # print("encoder", m.shape, p.shape)
        # visualize_channels(m[0], 4, 2, "m0")
        # visualize_channels(graph_out[0], 4, 2, "g0")
        # print(m.shape, out.shape)
        out = self.resnet.decoder(out)
        # print(graph_out[0].shape, self.m_coefs[0][4].shape)
        for i in range(self.num_layers):
            # m, p = m + graph_out[self.num_layers - 1 - i], p + graph_out[self.num_layers - 1 - i]
            # print(m.shape, p.shape, self.m_coefs[self.num_layers-1-i][2].shape, self.m_coefs[self.num_layers-i-2][3].shape) if i == 0 else None
            # m, p = self.Decoder[i](m, p, [self.m_coefs[self.num_layers-i][2], self.m_coefs[self.num_layers-i-1][3]],
            #                        [self.p_coefs[self.num_layers-i][2], self.p_coefs[self.num_layers-i-1][3]]) \
            #     if self.attn_mode[i + self.num_layers] else self.Decoder[i](m, p)
            # m, p = self.Decoder[i](m, p, [self.m_coefs[0][2]], [self.p_coefs[0][2]])
            m, p = self.Decoder[i](m, p, [self.m_coefs[0][2], self.conv_mg(graph_out[0])],
                                   [self.p_coefs[0][2], self.conv_pg(graph_out[1])])
            
            # m, p = self.Decoder[i](m, p, [self.m_coefs[0][2], self.m_coefs[0][3]],
            #                        [self.p_coefs[0][2], self.p_coefs[0][3]])
            # visualize_channels(m.transpose(1,2).view(B, -1, H, W), 32, 4, f'{i}_m_decode.png')
            # visualize_channels(p.transpose(1,2).view(B, -1, H, W), 32, 4, f'{i}_p_decode.png')
            m = m + m_list[self.num_layers - 1 - i]
            p = p + p_list[self.num_layers - 1 - i]
            # visualize_channels(m.transpose(1,2).view(B, -1, H, W), 32, 4, f'{i}_m_out.png')
            # visualize_channels(p.transpose(1,2).view(B, -1, H, W), 32, 4, f'{i}_p_out.png')
        # x = self.norm(m + p)
        x = self.head(m + p)
        # x = m + p
        x = x.view(B, self.embed_dim, H // self.patch_size, W // self.patch_size)
        # visualize_channels((m+p).transpose(1,2).view(B, -1, H, W), 32, 4, f'trans_out.png')
        final = self.forward_feature2img(x, hm) + out
        # xianhua(self.forward_feature2img(x, hm)[0], 'xianhuatrans.png')
        # visualize_channels(out, 4, 2, f'cnn_out.png')
        # xianhua(out[0], 'xianhuacnn.png')
        # visualize_channels(final, 4, 2, f'finalout.png')
        # xianhua(final[0], 'xianhuafinal.png')
        # x = self.pixel_shuffle(x)
        # x = self.convimg(x)
        # final = torch.clamp(final, 0, 1)
        return final



def Net(args):
    model = Graphsharpening(img_size=args['patch_size'],
                            in_chans=args['num_channels'],
                            patch_size=args['edtrans']['patch_size'],
                            window_size=args['edtrans']['window_size'],
                            embed_dim=args['edtrans']['embed_dim'],
                            depths=args['edtrans']['depths'],
                            num_heads=args['edtrans']['num_heads'],
                            attn_mode=args['edtrans']['attn_mode'],
                            exc_mode=args['edtrans']['exc_mode'],
                            device=args['device']
                            )
    return model


if __name__ == '__main__':
    device = 'cuda:0'
    ms = torch.randn([1, 4, 32, 32]).to(device)
    pan = torch.randn([1, 1, 128, 128]).to(device)
    args = {
            'num_channels': 4,
            'patch_size': 32,
            'device': device,
            "edtrans": {
                "img_size": 32,
                "patch_size": 1,
                "window_size": 16,
                "embed_dim": 16,
                "drop_rate": 0,
                'attn_mode': [0, 0, 1, 1],
                'exc_mode': [1, 1, 0, 0],
                "depths": [2, 2, 2, 2],
                "num_heads": [4, 4, 4, 4]
                # 'attn_mode': [0, 1],
                # 'exc_mode': [1, 0],
                # "depths": [2, 2],
                # "num_heads": [4, 4]
                # "depths": [2, 2],
                # "num_heads": [4, 4]
                # "depths": [2, 4],
                # "num_heads": [3, 6]
            }
    }
    module = Net(args).to(device)
    # print(module)
    import thop
    flops, params = thop.profile(module, inputs=(ms, pan))
    print('flops:', flops)
    print('params:', params)
    result = module(ms, pan)
    print(result.shape)



# if __name__ == '__main__':
#     device = 'cuda:1'
#     m = torch.randn([20, 4, 32, 32]).to(device)
#     p = torch.randn([20, 1, 128, 128]).to(device)
#     args = {
#         'num_channels': 4,
#         'patch_size': 32,
#         'device': device,
#         "edtrans": {
#             "img_size": 32,
#             "patch_size": 4,
#             "window_size": 8,
#             "embed_dim": 64,
#             "drop_rate": 0,
#             'attn_mode': [0, 0, 0, 0],
#             'exc_mode': [0, 0, 0, 0],
#             "depths": [2, 2, 2, 2],
#             "num_heads": [4, 4, 4, 4]
#             # "depths": [2, 2, 4],
#             # "num_heads": [3, 3, 6]
#             # "depths": [2, 4],
#             # "num_heads": [3, 6]
#         }
#     }
#     model = Net(args).to(device)
#     y = model(m, p)
#     print(y.shape)
