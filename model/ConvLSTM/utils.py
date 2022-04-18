import numpy as np
import pathlib
import torch.nn.functional as F
from torch.utils.data import Dataset
import glob
import os


def load_ndvi(full_data_path, sidx, eidx):
    files = glob.glob(os.path.join(full_data_path, "ndvi*.npy"))
    data = np.array([np.load(f) for f in sorted(files)[sidx:eidx]])
    return data

def load_precip(full_data_path, sidx, eidx):
    files = glob.glob(os.path.join(full_data_path, "precip_????_???_*.npy"))
    data = np.array([np.load(f) for f in sorted(files)[sidx:eidx]])
    return data

def prepare_inputs_targets(len_time, input_gap, input_length, pred_shift, pred_length, samples_gap):
    """
    Args:
        input_gap=1: time gaps between two consecutive input frames
        input_length=12: the number of input frames
        pred_shift=26: the lead_time of the last target to be predicted
        pred_length=26: the number of frames to be predicted
        samples_gap: the gap between the starting time of two retrieved samples
    Returns:
        idx_inputs: indices pointing to the positions of input samples
        idx_targets: indices pointing to the positions of target samples
    """
    assert pred_shift >= pred_length
    input_span = input_gap * (input_length - 1) + 1
    pred_gap = pred_shift // pred_length
    input_ind = np.arange(0, input_span, input_gap)
    target_ind = np.arange(0, pred_shift, pred_gap) + input_span + pred_gap - 1
    ind = np.concatenate([input_ind, target_ind]).reshape(1, input_length + pred_length)
    max_n_sample = len_time - (input_span + pred_shift - 1)
    ind = ind + np.arange(max_n_sample)[:, np.newaxis] @ np.ones((1, input_length + pred_length), dtype=int)
    idx_inputs = ind[::samples_gap, :input_length]
    idx_targets = ind[::samples_gap, input_length:]
    return idx_inputs, idx_targets

def unfold_StackOverChannel(img, kernel_size):
    """
    patch the image and stack individual patches along the channel
    Args:
        img (N, *, C, H, W): the last two dimensions must be the spatial dimension
        kernel_size: tuple of length 2
    Returns:
        output (N, *, C*H_k*N_k, H_output, W_output)
    """
    n_dim = len(img.size())
    assert n_dim == 4 or n_dim == 5
    if kernel_size[0] == 1 and kernel_size[1] == 1:
        return img

    pt = img.unfold(-2, size=kernel_size[0], step=kernel_size[0])
    pt = pt.unfold(-2, size=kernel_size[1], step=kernel_size[1]).flatten(-2)  # (N, *, C, n0, n1, k0*k1)
    if n_dim == 4:  # (N, C, H, W)
        pt = pt.permute(0, 1, 4, 2, 3).flatten(1, 2)
    elif n_dim == 5:  # (N, T, C, H, W)
        pt = pt.permute(0, 1, 2, 5, 3, 4).flatten(2, 3)
    assert pt.size(-3) == img.size(-3) * kernel_size[0] * kernel_size[1]
    return pt


def fold_tensor(tensor, output_size, kernel_size):
    """
    reconstruct the image from its non-overlapping patches
    Args:
        input tensor shape (N, *, C*k_h*k_w, n_h, n_w)
        output_size: (H, W), the size of the original image to be reconstructed
        kernel_size: (k_h, k_w)
        note that the stride is usually equal to kernel_size for non-overlapping sliding window
    Returns:
        output (N, *, C, H=n_h*k_h, W=n_w*k_w)
    """
    n_dim = len(tensor.size())
    assert n_dim == 4 or n_dim == 5
    if kernel_size[0] == 1 and kernel_size[1] == 1:
        return tensor
    f = tensor.flatten(0, 1) if n_dim == 5 else tensor
    folded = F.fold(f.flatten(-2), output_size=output_size, kernel_size=kernel_size, stride=kernel_size)
    if n_dim == 5:
        folded = folded.reshape(tensor.size(0), tensor.size(1), *folded.size()[1:])
    return folded

    
class ndviDataset(Dataset):
    def __init__(self, full_data_path, start_time_idx, end_time_idx, input_gap,
                 input_length, pred_shift, pred_length, samples_gap):
        """
        Args:
            full_data_path: the path specifying where the processed data file is located
            start_time/end_time: used to specify the dataset (train/eval/test) period
            sie_mask_period: the time period used to find the grid cells where the sea ice has ever appeared,
                             this is used to prevent the model attending to open sea area during training,
                             and also used during evaluation to better evaluate model performance
        """
        super().__init__()
        
        #load ndvi
        data = load_ndvi(full_data_path, start_time_idx, end_time_idx)
        masks = np.where(data > -1999, 1, 0).astype(np.float32)

        # load precip 
        data2 = load_precip(full_data_path, start_time_idx, end_time_idx)

        #TBD
        #try out other feature scaling 
        data = np.where(data > -1999, data, 0)*1e-4
    
        ##TBD
        #scaling precip??

        idx_inputs, idx_targets = prepare_inputs_targets(data.shape[0],
                     input_gap=input_gap, input_length=input_length,
                     pred_shift=pred_shift, pred_length=pred_length, 
                     samples_gap=samples_gap)
        


        self.train_masks = masks[idx_targets][:, :, None]
        
        #self.inputs = data[idx_inputs][:, :, None]
        ndvi = data[idx_inputs][:, :, None]
        pr = data[idx_inputs][:, :, None]
        self.inputs = np.concatenate((ndvi, pr), axis=2)

        self.targets = data[idx_targets][:, :, None]


    def getDataShape(self):
        return {'train_masks' : self.train_masks.shape,
                'inputs' : self.inputs.shape,
                'targets' : self.targets.shape}

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index], self.train_masks[index]