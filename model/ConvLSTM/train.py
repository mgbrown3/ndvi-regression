import torch
from config import configs
from trainer import Trainer
from utils import ndviDataset

if __name__ == '__main__':
    start_train, end_train = configs.train_period
    start_eval, end_eval = configs.eval_period
    input_gap = configs.input_gap
    input_length = configs.input_length
    pred_shift = configs.pred_shift
    output_length = configs.output_length

    print(f'loading train dataset from {start_train} to {end_train}')
    # data_path
    full_data_path = "/att/nobackup/jli30/workspace/ndvi_MGBrown_notebooks/data/timeseries"
    dataset_train = ndviDataset(full_data_path, start_train, end_train,
                                input_gap, input_length, pred_shift, output_length,
                                samples_gap=1)
    print(dataset_train.getDataShape())


    print(f'loading eval dataset from {start_eval} to {end_eval}')
    dataset_eval = ndviDataset(full_data_path, start_eval, end_eval,
                               input_gap, input_length, pred_shift, output_length,
                               samples_gap=1)
    print(dataset_eval.getDataShape())

    trainer = Trainer(configs)

    trainer.train(dataset_train, dataset_eval, 'checkpoint.chk')