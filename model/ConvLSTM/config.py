import torch

class Configs:
    def __init__(self):
        pass

configs = Configs()

# trainer related
configs.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
configs.batch_size_test = 5
configs.batch_size = 1
configs.lr = 0.001
configs.weight_decay = 0
configs.display_interval = 50
configs.num_epochs = 100
configs.early_stopping = True
configs.patience = 5
configs.gradient_clipping = True
configs.clipping_threshold = 1.

# patch
configs.patch_size = (2, 2)
configs.img_size = (52, 75)

# data related 
configs.input_dim = 2
configs.output_dim = 1

configs.input_length = 6
configs.output_length = 1

configs.input_gap = 1
configs.pred_shift = 1

configs.train_period = (0, 39)
configs.eval_period = (33, 46)

# model related
configs.kernel_size = (3, 3)
configs.bias = True
configs.hidden_dim = (96, 96, 96, 96)