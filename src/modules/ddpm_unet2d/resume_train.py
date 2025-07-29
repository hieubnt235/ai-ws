import os

import torch
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig, LRSchedulerConfigType, OptimizerLRScheduler
from torch.optim.optimizer import ParamsT

if os.getenv("LOG_FUNC", None) is None:
    # os.environ["LOG_FUNC"]= '1'
    pass

import lightning as L
from data.huggan_smithsonian_butterflies_subset import (
    DataModule,
    DataModuleConfig,
    DataLoaderConfig,
)
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    TQDMProgressBar,
)

from modules.ddpm_unet2d import (
    DDPMUNet2D,
    Metric,
    DDPMUNet2DConfig,
    TrainConfig,
)

dirpath = "ddpm_huggan_smithsonian_butterflies_subset_resume"
checkpoints_path = f"{dirpath}/checkpoints"

def create_metric_checkpoint_callback(metric: Metric):
    return ModelCheckpoint(
        dirpath=checkpoints_path,
        filename=f"{{epoch}}-{{step}}-min_{{{metric}:.4f}}",
        monitor=metric,
        mode="min",
        save_top_k=1,
        save_on_train_epoch_end=True,
    )

def optimizer_factory(p: ParamsT, lr=3e-6, **kwargs) -> OptimizerLRScheduler:
    optimizer = torch.optim.AdamW(p, lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.3,
        patience=3,
        eps=0
    )
    return OptimizerLRSchedulerConfig(
        optimizer=optimizer,
        lr_scheduler=LRSchedulerConfigType(
            scheduler=lr_scheduler,
            name="resume_optimizer",
            # Updating frequency of the scheduler ( call scheduler.step())
            interval="epoch",
            frequency=1,
            # The value for scheduler to depend on to update
            monitor=Metric.TRAIN_LOSS,
            strict=True,  # Must have monitor value
        ),
    )



if __name__ == "__main__":
    batchsize = 32
    datamodule = DataModule(
        config=DataModuleConfig(
            train_ratio=0.9,
            train_dl_cfg=DataLoaderConfig(batchsize=batchsize, num_workers=20),
            val_dl_cfg=DataLoaderConfig(batchsize=batchsize, num_workers=20),
        )
    )

    pretrained_model: DDPMUNet2D = DDPMUNet2D.load_from_checkpoint(
        "/home/a3ilab01/h-ws/ddpm_huggan_smithsonian_butterflies_subset/checkpoints/epoch=74-step=1125-min_train_loss=0.0168.ckpt",
        config=DDPMUNet2DConfig(
            train=TrainConfig(
                val_fid_gen_image_n_steps=50,
                optimizers_factory=optimizer_factory
            )
        )
    )

    assert pretrained_model.config.train.optimizers_factory == optimizer_factory
    assert pretrained_model.configure_optimizers()["lr_scheduler"]["scheduler"].patience == 3

    callbacks = [create_metric_checkpoint_callback(metric) for metric in pretrained_model.metrics]
    callbacks.extend(
        [
            ModelCheckpoint(
                dirpath=checkpoints_path,
                filename="{epoch}-{step}-last",
                save_top_k=1,
                save_on_train_epoch_end=True,
            ),
            TQDMProgressBar(leave=True),
        ]
    )

    loggers = [
        pl_loggers.CSVLogger(dirpath, version=0),
    ]

    trainer = L.Trainer(
        default_root_dir=dirpath,
        max_epochs=1000,
        callbacks=callbacks,
        logger=loggers,
        num_sanity_val_steps=0,
    )

    trainer.fit(
        model=pretrained_model,
        datamodule=datamodule,
    )
