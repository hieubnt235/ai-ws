import os

if os.getenv("LOG_FUNC", None) is None:
    # os.environ["LOG_FUNC"]= '1'
    pass

import lightning as L
from ds_utils.huggan_smithsonian_butterflies_subset import (
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

batchsize = 32
datamodule = DataModule(
    config=DataModuleConfig(
        train_ratio=0.9,
        train_dl_cfg=DataLoaderConfig(batchsize=batchsize, num_workers=20),
        val_dl_cfg=DataLoaderConfig(batchsize=batchsize, num_workers=20),
    )
)
# noinspection PyTypeChecker
model = DDPMUNet2D(
    config=DDPMUNet2DConfig(train=TrainConfig(val_fid_gen_image_n_steps=50))
)
dirpath = "ddpm_huggan_smithsonian_butterflies_subset"
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


callbacks = [create_metric_checkpoint_callback(metric) for metric in model.metrics]
callbacks.extend(
    [
        ModelCheckpoint(
            dirpath=checkpoints_path,
            filename="{epoch}-{step}-last",
            save_top_k=1,
            save_on_train_epoch_end=True,
        ),
        # RichProgressBar(leave=True),
        TQDMProgressBar(leave=True),
    ]
)

loggers = [
    pl_loggers.CSVLogger(dirpath, version=0),
    # pl_loggers.TensorBoardLogger(dirpath, version=0),
]

trainer = L.Trainer(
    default_root_dir=dirpath,
    max_epochs=1000,
    callbacks=callbacks,
    logger=loggers,
    num_sanity_val_steps=0,
)

trainer.fit(
    model=model,
    datamodule=datamodule,
)
