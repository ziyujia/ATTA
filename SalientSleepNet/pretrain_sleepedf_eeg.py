import time
from argparse import ArgumentParser
from functools import reduce
from typing import Dict, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import optim
from torch.nn import functional as fn
from torch.utils.data import DataLoader

from model_modified import SalientSleepNet_without_up
from preprocess_sleepedf import get_double_eeg_datasets


class ModelWrapper(pl.LightningModule):
    def __init__(self, window_stride: int = 35):
        super(ModelWrapper, self).__init__()
        self.ssn = SalientSleepNet_without_up(window_stride)

        self.train_acc = torchmetrics.Accuracy()
        self.train_f1 = torchmetrics.F1Score(
            num_classes=5,
            average='macro',
            mdmc_average='global'
        )
        self.val_acc = torchmetrics.Accuracy()
        self.val_f1 = torchmetrics.F1Score(
            num_classes=5,
            average='macro',
            mdmc_average='global'
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ssn(x)

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_index) -> Dict:
        x, y = train_batch
        preds = self.ssn(x)
        loss = fn.cross_entropy(
            preds, y.long(),
            weight=torch.Tensor((1.0, 1.80, 1.0, 1.25, 1.20)).cuda()
        )
        self.log("train_step_loss", loss)
        return {'loss': loss, 'preds': preds.detach(), 'target': y}

    def training_step_end(self, outputs):
        self.train_acc.update(outputs['preds'], outputs['target'])
        self.train_f1.update(outputs['preds'], outputs['target'])
        self.log("train_acc", self.train_acc.compute())
        self.log("train_f1", self.train_f1.compute())

    def training_epoch_end(self, outputs) -> None:
        print(f"\nTraining Accuracy: {self.train_acc.compute():.4f}\t\
                    Training F1: {self.train_f1.compute():.4f}")
        self.log("train_acc", self.train_acc.compute())
        self.log("train_f1", self.train_f1.compute())
        self.train_acc.reset()
        self.train_f1.reset()

    def validation_step(self, val_batch, batch_index) -> Dict:
        x, y = val_batch
        preds = self.ssn(x)
        loss = fn.cross_entropy(preds, y.long())
        self.log("val_step_loss", loss)
        return {'loss': loss, 'preds': preds.detach(), 'target': y}

    def validation_step_end(self, outputs):
        self.val_acc.update(outputs['preds'], outputs['target'])
        self.val_f1.update(outputs['preds'], outputs['target'])
        self.log("val_acc", self.val_acc.compute())
        self.log("val_f1", self.val_f1.compute())

    def validation_epoch_end(self, outputs) -> None:
        print(f"\nValidation Accuracy: {self.val_acc.compute():.4f}\t\
                            Validation F1: {self.val_f1.compute():.4f}")
        self.log("train_acc", self.val_acc.compute())
        self.log("train_f1", self.val_f1.compute())
        self.val_acc.reset()
        self.val_f1.reset()


def parse_arguments() -> Tuple[str, int, int, int, int, str, str]:
    parser = ArgumentParser()
    parser.add_argument(
        '--data_dir', '-d', type=str,
        default='../../sleep_data/sleepedf-153'
    )
    parser.add_argument('--batch_size', '-b', type=int, default=10)
    parser.add_argument('--train_epoch', '-t', type=int, default=50)
    parser.add_argument('--folds', '-f', type=int, default=5)
    parser.add_argument("--window_size", "-w", type=int, default=35)
    parser.add_argument("--pth_dir", "-p", type=str, default=r'pretrained_model/model_sleepedf_eeg.pth')
    parser.add_argument('--logger_name', type=str, default=None)
    args = parser.parse_args()
    return (
        args.data_dir,
        args.batch_size,
        args.train_epoch,
        args.folds,
        args.window_size,
        args.pth_dir,
        args.logger_name
    )


if __name__ == '__main__':
    data_path1, batch_size, epoch, folds, window_size, pth_dir, logger_name = parse_arguments()
    # Dataset
    dataset_list = get_double_eeg_datasets(data_path1,stride=window_size)

    assert len(dataset_list) > folds, "\
        too less data for training and validation"
    # Normalize the dataset.
    for dataset in dataset_list:
        dataset.normalization()
    # Divide the dataset into multiple fold.
    split_data_list = np.array_split(
        np.asarray(dataset_list, dtype=object), folds
    )

    if logger_name is None:
        logger_name = f"ssn-{time.strftime('%y-%m-%d_%H_%M', time.localtime())}"
    # k-fold cross validation
    for k in range(folds):
        print(f"\n=======================================\
            ✨ fold: {k + 1} / {folds} \
            ===============================================")
        # train set
        train_set = list(set(dataset_list) - set(split_data_list[k]))
        # validation set
        val_set = split_data_list[k]
        train_set = reduce(lambda x, y: x + y, train_set)
        val_set = reduce(lambda x, y: x + y, val_set)
        train_loader = DataLoader(
            train_set, batch_size=batch_size,
            shuffle=True, num_workers=4, drop_last=False
        )
        val_loader = DataLoader(
            val_set, batch_size=batch_size,
            shuffle=False, num_workers=4, drop_last=False
        )
        logger = TensorBoardLogger(
            f'./lightning_logs/{logger_name}',
            name=f'fold{k+1}'
        )

        model = ModelWrapper(window_size)
        # early stopping
        early_stop_callback = EarlyStopping(
            monitor='train_f1', mode="max", patience=5
        )
        model_checkpointer_callback = ModelCheckpoint(
            monitor='val_f1', mode='max',
            filename=f'ssn-fold{k + 1}-{{epoch:02d}}-{{val_f1:.2f}}'
        )

        trainer = pl.Trainer(
            logger=logger,
            gpus=1, max_epochs=epoch,
            callbacks=[early_stop_callback, model_checkpointer_callback]
        )
        trainer.fit(model, train_loader, val_loader)
        torch.save(model.state_dict(), pth_dir)