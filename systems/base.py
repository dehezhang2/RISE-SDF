import pytorch_lightning as pl

import models
from systems.utils import parse_optimizer, parse_scheduler, update_module_step
from utils.mixins import SaverMixin
from utils.misc import config_to_primitive, get_rank
import torch

class BaseSystem(pl.LightningModule, SaverMixin):
    """
    Two ways to print to console:
    1. self.print: correctly handle progress bar
    2. rank_zero_info: use the logging module
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rank = get_rank()
        self.prepare()
        self.model = models.make(self.config.model.name, self.config.model)

    def prepare(self):
        pass

    def forward(self, batch):
        raise NotImplementedError

    def C(self, value):
        if isinstance(value, int) or isinstance(value, float):
            pass
        else:
            value = config_to_primitive(value)
            if not isinstance(value, list):
                raise TypeError('Scalar specification only supports list, got', type(value))
            if len(value) == 3:
                value = [0] + value
            assert len(value) == 4
            start_step, start_value, end_value, end_step = value
            if isinstance(end_step, int):
                current_step = self.global_step
                value = start_value + (end_value - start_value) * max(min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0)
            elif isinstance(end_step, float):
                current_step = self.current_epoch
                value = start_value + (end_value - start_value) * max(min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0)
        return value

    def preprocess_data(self, batch, stage):
        pass

    def on_train_start(self) -> None:
        self.dataset = self.trainer.datamodule.train_dataloader().dataset

    def on_validation_start(self) -> None:
        self.dataset = self.trainer.datamodule.val_dataloader().dataset
        if hasattr(self.model, "emitter"):
            env_map = self.model.emitter.generate_image()
            # for i in range(env_map.shape[0]):
            self.save_image_grid(f"it{self.global_step}-envmap.exr", [
                {'type': 'hdr', 'img': env_map, 'kwargs': {'data_format': 'HWC'}},
            ])

    def on_test_start(self) -> None:
        self.dataset = self.trainer.datamodule.test_dataloader().dataset
        if hasattr(self.model, "emitter"):
            env_map = self.model.emitter.generate_image()
            self.save_image_grid(f"it{self.global_step}-envmap.exr", [
                {'type': 'hdr', 'img': env_map, 'kwargs': {'data_format': 'HWC'}},
            ])
    def on_predict_start(self) -> None:
        self.dataset = self.trainer.datamodule.predict_dataloader().dataset
        if hasattr(self.model, "emitter"):
            env_map = self.model.emitter.generate_image()
            self.save_image_grid(f"it{self.global_step}-envmap.exr", [
                {'type': 'hdr', 'img': env_map, 'kwargs': {'data_format': 'HWC'}},
            ])

    """
    Implementing on_after_batch_transfer of DataModule does the same.
    But on_after_batch_transfer does not support DP.
    """
    def on_train_batch_start(self, batch, batch_idx, unused=0):
        self.dataset = self.trainer.datamodule.train_dataloader().dataset
        update_module_step(self.model, self.current_epoch, self.global_step)
        self.preprocess_data(batch, 'train')
        # update_module_step(self.model, self.current_epoch, self.global_step)
        # Re-create optimizer and scheduler when upsampling grids
        if hasattr(self.model, 'upsample_milestones'):
            if self.global_step in self.model.upsample_milestones:
                # ret = self.configure_optimizers()
                # self.optimizers = ret["optimizer"]
                # self.lr_schedulers = ret["lr_scheduler"]
                self.trainer.strategy.setup_optimizers(self.trainer)
                print("Re-create optimizer and scheduler at step", self.global_step)
        if hasattr(self, "reinit_optimizer_steps"):
            if self.step in self.reinit_optimizer_steps:
                self.trainer.strategy.setup_optimizers(self.trainer)
                print("\nRe-create optimizer and scheduler at step", self.global_step)

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        self.dataset = self.trainer.datamodule.val_dataloader().dataset
        self.preprocess_data(batch, 'validation')
        update_module_step(self.model, self.current_epoch, self.global_step)

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx):
        self.dataset = self.trainer.datamodule.test_dataloader().dataset
        self.preprocess_data(batch, 'test')
        update_module_step(self.model, self.config.trainer.max_steps / 100, self.config.trainer.max_steps)
        

    def on_predict_batch_start(self, batch, batch_idx, dataloader_idx):
        self.dataset = self.trainer.datamodule.predict_dataloader().dataset
        self.preprocess_data(batch, 'predict')
        update_module_step(self.model, self.current_epoch, self.global_step)

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    """
    # aggregate outputs from different devices (DP)
    def training_step_end(self, out):
        pass
    """

    """
    # aggregate outputs from different iterations
    def training_epoch_end(self, out):
        pass
    """

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    """
    # aggregate outputs from different devices when using DP
    def validation_step_end(self, out):
        pass
    """

    def validation_epoch_end(self, out):
        """
        Gather metrics from all devices, compute mean.
        Purge repeated results using data index.
        """
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_epoch_end(self, out):
        """
        Gather metrics from all devices, compute mean.
        Purge repeated results using data index.
        """
        raise NotImplementedError

    def export(self):
        raise NotImplementedError

    def configure_optimizers(self):
        if hasattr(self, "reinit_optimizer_steps") and self.step in self.reinit_optimizer_steps:
            del self.config.system.optimizer.params['variance']
            del self.config.system.optimizer.params['emitter']
            del self.config.system.optimizer.params['material']
            del self.config.system.optimizer.params['geometry']
            # del self.config.system.optimizer.params['albedo']
            del self.config.system.optimizer.params['texture']
            print("\nremoviong the frozen parameters from optimizer")
        optim = parse_optimizer(self.config.system.optimizer, self.model)
        ret = {
            'optimizer': optim,
        }
        if 'scheduler' in self.config.system:
            ret.update({
                'lr_scheduler': parse_scheduler(self.config.system.scheduler, optim),
            })
        return ret
