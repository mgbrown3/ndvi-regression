import torch.nn as nn
from convlstm_net import ConvLSTM
from utils import unfold_StackOverChannel, fold_tensor

class NdviNet(nn.Module):
    def __init__(self, configs):
        super().__init__()
        input_dim = configs.input_dim*configs.patch_size[0]*configs.patch_size[1]
        output_dim = configs.output_dim*configs.patch_size[0]*configs.patch_size[1]

        self.base_net = ConvLSTM(input_dim, configs.hidden_dim, output_dim, configs.kernel_size)

        self.patch_size = configs.patch_size
        self.img_size = configs.img_size

    def forward(self, x):
        y = self.base_net(unfold_StackOverChannel(x, kernel_size=self.patch_size))
        y = fold_tensor(y, output_size=self.img_size, kernel_size=self.patch_size)

        return y