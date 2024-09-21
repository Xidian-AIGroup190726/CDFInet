import torch
from model.cgfi_n import Net  # Our method is saved at /model/cgfi_n

if __name__ == '__main__':
    # There is a demo about running our model.

    device = 'cuda:0'
    # m_path = '../../../Remote data/PSharpen/mat/6 WorldView-3/MS_256/1.mat'
    # p_path = '../../../Remote data/PSharpen/mat/6 WorldView-3/PAN_1024/1.mat'
    # import scipy.io
    # ms = np.expand_dims(np.transpose(scipy.io.loadmat(m_path)['imgMS'], (2, 0, 1)), axis=0)
    # pan = np.expand_dims(np.expand_dims(scipy.io.loadmat(p_path)['imgPAN'], axis=0), axis=0)
    # ms = torch.from_numpy(to_tensor(ms[:, :4, :32, :32])).type(torch.FloatTensor).to(device)
    # pan = torch.from_numpy(to_tensor(pan[:, :, :128, :128])).type(torch.FloatTensor).to(device)

    ms = torch.randn([1, 4, 32, 32]).to(device)
    pan = torch.randn([1, 1, 128, 128]).to(device)
    args = {
        'num_channels': 4,
        'patch_size': 32,
        'device': device
    }
    module = Net(args).to(device)
    result = module(ms, pan)
    print(result.shape)   # torch.size([B, C, H, W])