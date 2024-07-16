import sys
from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MinMetric
from torchmetrics.regression.mae import MeanAbsoluteError
from torchmetrics.regression.mse import MeanSquaredError
from torchmetrics.regression.r2 import R2Score


class SEResNet50LitModule(LightningModule):
    """Example of LightningModule for Squeeze Excite Residual Network (SEResNet50) for regression.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ):
        """

        Args:
            net:
            optimizer:
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net

        # loss function
        self.criterion_mse = torch.nn.MSELoss()
        self.criterion_l1 = torch.nn.L1Loss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()

        # for logging best so far validation MSE
        self.val_mse_best = MinMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_mse_best doesn't store MSE from these checks
        self.val_mse_best.reset()

    def step(self, batch: Any):
        x, y = batch
        B = x.shape[0]
        preds = torch.tanh(self.forward(x))
        loss_mse = self.criterion_mse(preds, y)

        l1_preds = x * preds[:, 0].view(B, 1, 1, 1, 1) + preds[:, 1].view(B, 1, 1, 1, 1)
        a = y[:, 0].view(B, 1, 1, 1, 1)
        b = y[:, 1].view(B, 1, 1, 1, 1)

        l1_targets = x * a + b
        loss_l1 = self.criterion_l1(l1_preds, l1_targets)

        loss = loss_mse + loss_l1

        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        mse = self.train_mse(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train/mse", mse, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def on_train_epoch_end(self,):
        # `outputs` is a list of dicts returned from `training_step()`
        self.train_mse.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        mse = self.val_mse(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/mse", mse, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def on_validation_epoch_end(self):
        mse = self.val_mse.compute()  # get val MSE from current epoch
        self.val_mse_best.update(mse)
        self.log("val/mse_best", self.val_mse_best.compute(), on_epoch=True, prog_bar=True)
        self.val_mse.reset()

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        mse = self.test_mse(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/mse", mse, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def on_test_epoch_end(self):
        self.test_mse.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return {
            "optimizer": self.hparams.optimizer(params=self.parameters()),
        }


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "mnist.yaml")
    _ = hydra.utils.instantiate(cfg)
