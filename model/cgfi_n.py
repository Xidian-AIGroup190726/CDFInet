import torch.nn as nn
import torch
from model.panform.common.modules import conv3x3, SwinModule
from model.contourlet_torch import ContourDec, lpdec_layer, dfbdec_layer
from model.gconvtrans import Graph2dConvolution
from torchvision import transforms
from model.gcnn import GCN, conv3x3, BasicBlk
import numpy as np

class ContourN(nn.Module):
    def __init__(self, dir=2):
        super().__init__()
        self.dir = dir
        
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
        
    def forward(self, img):
        B, C, H, W = img.shape
        l, h = lpdec_layer(img)
        dh = self.dfb(h)
        return [h, dh, l]
    

class GCTN(nn.Module):
    def __init__(self, in_chan, chan_num, bl_num=4):
        super(GCTN, self).__init__()
        self.gc = Graph2dConvolution(in_chan, chan_num, kernel_size=3, block_num=chan_num, padding=1)
        self.convin = nn.Conv2d(in_chan, in_chan, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(chan_num)

    def forward(self, x, index_f=None):
        convin = self.convin(index_f) if index_f is not None else self.convin(x)
        # index_f = lpdec_layer(x)[1]
        value, index = torch.max(convin, dim=1, keepdim=True)
        # print(value, index)
        x = self.gc(x, index)
        x = self.bn(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, inplanes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.max_pool = nn.MaxPool2d(1)
        self.avg_pool = nn.AvgPool2d(1)
        # 通道注意力，即两个全连接层连接
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=inplanes, out_channels=max(inplanes // ratio, 1), kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=max(inplanes // ratio, 1), out_channels=inplanes, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc(self.max_pool(x))
        avg_out = self.fc(self.avg_pool(x))
        # 最后输出的注意力应该为非负
        out = self.sigmoid(max_out + avg_out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, padding=7 // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 压缩通道提取空间信息
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # 经过卷积提取空间注意力权重
        x = torch.cat([max_out, avg_out], dim=1)
        out = self.conv1(x)
        # 输出非负
        out = self.sigmoid(out)  # torch.size([batch_size, 1, width, height])
        return out


class GEM(nn.Module):
    def __init__(self, in_chan, mode="h"):
        super().__init__()
        num_chan = in_chan if mode == 'h' else in_chan * 4
        self.ori = nn.Sequential(
            GCTN(in_chan, in_chan),
            nn.Conv2d(in_chan, in_chan, 1, 1),
            nn.BatchNorm2d(in_chan),
            nn.ReLU(inplace=True)
        )
        self.sa = ChannelAttention(num_chan)

        self.hdh = nn.Sequential(
            nn.Conv2d(in_chan, num_chan, 3, 1, 1),
            nn.BatchNorm2d(num_chan),
            nn.ReLU(inplace=True)
        )

        self.hf = GCTN(num_chan, num_chan)
        self.out = nn.Sequential(
            nn.Conv2d(num_chan, num_chan, 1, 1),
            nn.BatchNorm2d(num_chan),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, img, coefs):
        res = img
        out = self.ori(img) + res
        # out = torch.mul(self.sa(out), out) + res
        d_out = self.hdh(out)
        out = self.sa(self.hf(d_out, coefs)) * d_out
        return out
    

class DGEM(nn.Module):
    def __init__(self, in_chan, mode):
        super().__init__()
        self.in_chan = in_chan
        self.mh_GEM = GEM(in_chan, mode)
        self.ph_GEM = GEM(1, mode)

        self.num = 1 if mode == 'h' else 4

        self.gcn1 = GCTN((in_chan + 1)*self.num, (in_chan + 1)*self.num)
        self.sa = SpatialAttention()
        
    def forward(self, m, p, m_coefs, p_coefs):
        _, C, _, _ = m.shape
        m_out = self.mh_GEM(m, m_coefs)
        p_out = self.ph_GEM(p, p_coefs)
        p_dh1_expanded = p_coefs.repeat_interleave(C, dim=1)
        cross_index1 = torch.concat([m_coefs * p_dh1_expanded, p_coefs], dim=1)
        # from man import mutual_information
        # print('mp', mutual_information(m_out, p_out))

        out = torch.concat([m_out, p_out], dim=1)
        out = self.gcn1(out, cross_index1)
        attn_mask = self.sa(out)
        # m_cout, p_cout = torch.split(out, [self.in_chan*self.num, self.num], dim=1)
        # print('mp_c', mutual_information(m_cout, p_cout))
        # print('m_c', mutual_information(m_out, m_cout))
        # print('p_c', mutual_information(p_out, p_cout))
        # print('overall', mutual_information(m_out * attn_mask, p_out * (1-attn_mask)))
        return m_out * attn_mask, p_out * (1-attn_mask)
        

class Graph_enhance(nn.Module):
    def __init__(self, in_chan, num_chan):
        super().__init__()
        self.m_bk1_1 = GCTN(in_chan, in_chan, in_chan, in_chan)
        self.p_bk1_1 = GCTN(1, in_chan, in_chan, 1)

        self.m_bk1_2 = GCTN(in_chan, in_chan, in_chan, in_chan)
        self.p_bk1_2 = GCTN(in_chan, in_chan, in_chan, in_chan)

        self.m_bk2_1 = GCTN(in_chan, int(num_chan // 2), in_chan*4, in_chan*4)
        self.p_bk2_1 = GCTN(in_chan, int(num_chan // 2), in_chan*4, 4)
        self.gcn1 = GCTN(num_chan, num_chan, in_chan*4, in_chan*4)
        
    def forward(self, m, p, m_coefs, p_coefs):
        _, C, _, _ = m.shape
        m_out = self.m_bk1_1(m, m_coefs[0])
        p_out = self.p_bk1_1(p, p_coefs[0])
        
        m_out = self.m_bk2_1(m_out, m_coefs[1]) 
        p_out = self.p_bk2_1(p_out, p_coefs[1])
        
        p_dh1_expanded = p_coefs[1].repeat_interleave(C, dim=1)
        cross_index1 = m_coefs[1] * p_dh1_expanded
        
        out = torch.concat([m_out, p_out], dim=1)
        out = self.gcn1(out, cross_index1)
        m_cout, p_cout = torch.chunk(out, 2, 1)
        
        return m_out, p_out


class CNNnet(nn.Module):
    def __init__(self, block, num_blocks, args):
        super(CNNnet, self).__init__()
        in_chan = args['in_channels']
        num_channels = args['num_channels']
        self.in_planes = num_channels
        self.conv_in = conv3x3(1+in_chan, num_channels)
        self.BN = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, num_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, num_channels, num_blocks[1], stride=1)
        self.in_planes = num_channels+in_chan*4+4
        self.layer3 = self._make_layer(block, num_channels+in_chan*4+4, num_blocks[2], stride=1)
        self.layer4 = self._make_layer(block, num_channels+in_chan*4+4, num_blocks[3], stride=1)

        self.conv_out = conv3x3(num_channels+in_chan*4+4, in_chan)

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

class CrossTransformer(nn.Module):
    def __init__(self, ms_chans=4, n_feats=32, n_heads=4, head_dim=16, win_size=4,
                 n_blocks=1, cross_module=['pan', 'ms'], cat_feat=['pan', 'ms'], sa_fusion=False):
        super().__init__()
        # self.cfg = cfg
        self.n_blocks = n_blocks
        self.cross_module = cross_module
        self.cat_feat = cat_feat
        self.sa_fusion = sa_fusion
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')

        pan_encoder = [
            SwinModule(in_channels=1, hidden_dimension=n_feats, layers=2,
                       downscaling_factor=2, num_heads=n_heads, head_dim=head_dim,
                       window_size=win_size, relative_pos_embedding=True, cross_attn=False),
            SwinModule(in_channels=n_feats, hidden_dimension=n_feats, layers=2,
                       downscaling_factor=2, num_heads=n_heads, head_dim=head_dim,
                       window_size=win_size, relative_pos_embedding=True, cross_attn=False),
        ]
        ms_encoder = [
            SwinModule(in_channels=ms_chans, hidden_dimension=n_feats, layers=2,
                       downscaling_factor=2, num_heads=n_heads, head_dim=head_dim,
                       window_size=win_size, relative_pos_embedding=True, cross_attn=False),
            SwinModule(in_channels=n_feats, hidden_dimension=n_feats, layers=2,
                       downscaling_factor=2, num_heads=n_heads, head_dim=head_dim,
                       window_size=win_size, relative_pos_embedding=True, cross_attn=False),
        ]

        self.ms_cross_pan = nn.ModuleList()
        for _ in range(n_blocks):
            self.ms_cross_pan.append(SwinModule(in_channels=n_feats, hidden_dimension=n_feats, layers=2,
                                                downscaling_factor=1, num_heads=n_heads, head_dim=head_dim,
                                                window_size=win_size, relative_pos_embedding=True, cross_attn=True))


        self.pan_cross_ms = nn.ModuleList()
        for _ in range(n_blocks):
            self.pan_cross_ms.append(SwinModule(in_channels=n_feats, hidden_dimension=n_feats, layers=2,
                                                downscaling_factor=1, num_heads=n_heads, head_dim=head_dim,
                                                window_size=win_size, relative_pos_embedding=True, cross_attn=True))

        self.pan_encoder = nn.Sequential(*pan_encoder)
        self.ms_encoder = nn.Sequential(*ms_encoder)

    def forward(self, ms, pan, m_coefs, p_coefs):
        # ms = self.upsample(ms)
        pan_feat = self.pan_encoder(pan)
        ms_feat = self.ms_encoder(ms)

        last_pan_feat = pan_feat
        last_ms_feat = ms_feat
        for i in range(self.n_blocks):
            pan_cross_ms_feat = self.pan_cross_ms[i](last_pan_feat, y=last_ms_feat, dir=m_coefs)
            ms_cross_pan_feat = self.ms_cross_pan[i](last_ms_feat, y=last_pan_feat, dir=p_coefs)
            last_pan_feat = pan_cross_ms_feat
            last_ms_feat = ms_cross_pan_feat

        cat_list = []
        cat_list.append(last_pan_feat)
        cat_list.append(last_ms_feat)

        return torch.cat(cat_list, dim=1)


class pre_CGFI(nn.Module):
    def __init__(self):
        super().__init__()
        self.ct = ContourN()

    def forward(self, hm, pan):
        m_coefs, p_coefs = self.ct(hm), self.ct(pan)
        return m_coefs, p_coefs


class CGFI(nn.Module):
    def __init__(self, in_chans, embed_dim=16, num_heads=4, head_dim=16,
                 win_size=4, n_blocks=3, args=1):
        super().__init__()

        self.m_in = nn.Conv2d(in_chans*2, int(embed_dim//2), 1, 1)
        self.p_in = nn.Conv2d(2, int(embed_dim//2), 1, 1)
        self.BN = nn.BatchNorm2d(embed_dim)
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # self.l_graph_enhance = DGEM(in_chans, mode='h')
        self.h_graph_enhance = DGEM(in_chans, mode='h')
        self.dh_graph_enhance = DGEM(in_chans, mode='d')

        self.HF_Decoder = CNNnet(BasicBlk, [1, 1, 1, 1], {"in_channels": in_chans, "num_channels": embed_dim})
        self.LF_ContourTrans = CrossTransformer(in_chans, embed_dim)

        self.HR_tail = nn.Sequential(
            conv3x3(embed_dim * 2, embed_dim*4),
            nn.PixelShuffle(2), nn.ReLU(True), conv3x3(embed_dim, embed_dim * 4),
            nn.PixelShuffle(2), nn.ReLU(True), conv3x3(embed_dim, embed_dim),
            nn.ReLU(True), conv3x3(embed_dim, in_chans))
        # self.trans = former(args)

    def forward(self, hm, pan, m_coefs, p_coefs):
        B, C, H, W = hm.shape

        m_h, p_h = self.h_graph_enhance(hm, pan, m_coefs[0], p_coefs[0])
        m_dh, p_dh = self.dh_graph_enhance(m_h, p_h, m_coefs[1], p_coefs[1])

        m1, p1 = (self.m_in(torch.concat([hm, m_h], dim=1)),
                  self.p_in(torch.concat([pan, p_h], dim=1)))

        out = self.HF_Decoder(m1, p1)
        out = self.HF_Decoder.decoder(torch.concat([out, m_dh, p_dh], dim=1))

        out2 = self.LF_ContourTrans(hm - m_coefs[0], pan - p_coefs[0], m_dh, p_dh)
        out2 = self.HR_tail(out2)
        out = torch.clamp(out2, 0, 1) + out

        return out


class CGFI_n(nn.Module):
    def __init__(self, in_chans, embed_dim=16, num_heads=4, head_dim=16,
                 win_size=4, n_blocks=3, args=1):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

        self.CT = pre_CGFI()
        self.net = CGFI(in_chans=args['num_channels'], args=args)

    def forward(self, lms, pan):
        hm = self.upsample(lms)
        m_coefs, p_coefs = self.CT(hm, pan)

        out = self.net(hm, pan, m_coefs, p_coefs)
        return out


def Net(args):
    model = CGFI_n(in_chans=args['num_channels'], args=args)
    return model


def to_tensor(image):
    max_i = np.max(image)
    min_i = np.min(image)
    image = (image - min_i) / (max_i - min_i)
    return image


if __name__ == '__main__':
    device = 'cuda:1'
    m_path = '../../../Remote data/PSharpen/mat/6 WorldView-3/MS_256/1.mat'
    p_path = '../../../Remote data/PSharpen/mat/6 WorldView-3/PAN_1024/1.mat'
    import scipy.io
    ms = np.expand_dims(np.transpose(scipy.io.loadmat(m_path)['imgMS'], (2, 0, 1)), axis=0)
    pan = np.expand_dims(np.expand_dims(scipy.io.loadmat(p_path)['imgPAN'], axis=0), axis=0)
    ms = torch.from_numpy(to_tensor(ms[:, :4, :32, :32])).type(torch.FloatTensor).to(device)
    pan = torch.from_numpy(to_tensor(pan[:, :, :128, :128])).type(torch.FloatTensor).to(device)

    # ms = torch.randn([1, 8, 32, 32]).to(device)
    # pan = torch.randn([1, 1, 128, 128]).to(device)
    args = {
        'num_channels': 4,
        'patch_size': 32,
        'device': device
    }
    module = Net(args).to(device)
    result = module(ms, pan)
    print(result.shape)