import random
from typing import Sequence

import pytest
import torch
from diffusers import UNet2DModel, DDPMScheduler
import pickle
import lightning.pytorch.utilities.types as pl_types
from modules.ddpm_unet2d import (
    DDPMUNet2D,
    UNet2DConfig,
    SchedulerConfig,
    DDPMUNet2DConfig,
    Metric,
    TrainConfig,
)


@pytest.fixture(scope="module")
def unet2d():
    return UNet2DModel()


@pytest.fixture()
def scheduler():
    return DDPMScheduler()


@pytest.fixture(scope="module")
def ddpm_unet2d_config():
    return DDPMUNet2DConfig()


@pytest.fixture(scope="module")
def ddpm_unet2d(ddpm_unet2d_config):
    model = DDPMUNet2D(ddpm_unet2d_config)
    # if torch.cuda.is_available():
    #     return model.cuda()
    return model


def test_ddpm_unet2d_with_custom_unet_and_scheduler(unet2d, scheduler):
    # Random in_channels, which will be overridden later.
    a, b = 22, 33
    config = DDPMUNet2DConfig(
        unet=UNet2DConfig(in_channels=random.randint(a, b)),
        scheduler=SchedulerConfig(num_train_timesteps=random.randint(a, b)),
        train=TrainConfig(val_fid_gen_image_n_steps=a),
    )
    ddpm_unet2d = DDPMUNet2D(config, unet=unet2d, scheduler=scheduler)

    assert ddpm_unet2d.unet == unet2d
    assert ddpm_unet2d.scheduler == scheduler
    assert ddpm_unet2d.config.unet.in_channels == unet2d.config["in_channels"]
    assert (
        ddpm_unet2d.config.scheduler.num_train_timesteps
        == scheduler.config["num_train_timesteps"]
    )
    assert ddpm_unet2d.val_fid_gen_image_n_steps == a


def test_metrics_initialization(ddpm_unet2d):
    m_temp = []  # For testing that no metric exists more than one time.
    for metric in ddpm_unet2d.metrics:
        assert metric not in m_temp
        if metric in [Metric.TRAIN_LOSS, Metric.VAL_LOSS]:
            assert hasattr(ddpm_unet2d, metric)
        else:
            assert metric == Metric.VAL_FID
            assert hasattr(ddpm_unet2d, "fid_metric")
            assert hasattr(ddpm_unet2d, "all_real_loaded")
            assert isinstance((v := ddpm_unet2d.val_fid_gen_image_n_steps), int)
            assert v > 0
        m_temp.append(metric)


scale_factor = DDPMUNet2DConfig().unet._sample_scale_factor


@pytest.mark.parametrize(
    "noises",
    [
        torch.randn([3, scale_factor, scale_factor]),
        torch.randn([2, 3, scale_factor, scale_factor]),
    ],
)
@pytest.mark.parametrize("timesteps", [2, torch.tensor(2), 2.3])
def test_forward(ddpm_unet2d, noises, timesteps):
    with torch.inference_mode():
        output = ddpm_unet2d(noises, timesteps)
        assert len(output.shape) == 4

        if len(noises.shape) == 3:
            assert output.shape[1:] == noises.shape
        else:
            assert output.shape == noises.shape
            # 1-D timesteps case
            timesteps_1d = torch.randint(2, 5, [noises.size(0)])
            assert ddpm_unet2d(noises, timesteps_1d).shape == noises.shape


@pytest.mark.parametrize(
    "noise",
    [
        torch.randn([3, scale_factor, scale_factor]),
        torch.randn([2, 3, scale_factor, scale_factor]),
    ],
)
@pytest.mark.parametrize("clamp_output", [True, False, "center"])
def test_generate(ddpm_unet2d, noise, clamp_output):
    # noinspection PyTypeChecker
    gen = ddpm_unet2d.generate(noise, 2, clamp_output=clamp_output)
    assert len(gs := gen.shape) == 4
    if len(noise.shape) == 3:
        assert gs[1:] == noise.shape
    else:
        assert gs == noise.shape

    if clamp_output:
        if clamp_output == "center":
            mx, mn = 1.0, -1.0
        else:
            mx, mn = 1.0, 0.0
        assert (gmx := gen.max()) <= mx or torch.isclose(gmx, torch.tensor(mx))
        assert (gmn := gen.min()) >= mn or torch.isclose(gmn, torch.tensor(mn))


def test_loss_computation(ddpm_unet2d):
    images = torch.randn(2, 3, scale_factor, scale_factor).clamp(-1, 1)
    loss = ddpm_unet2d.calculate_loss(images)
    assert torch.is_tensor(loss)
    assert loss.shape == torch.Size([])


def test_pickle(ddpm_unet2d):
    byte_obj = pickle.dumps(ddpm_unet2d)
    loaded_ddpm_unet2d = pickle.loads(byte_obj)
    assert isinstance(loaded_ddpm_unet2d, DDPMUNet2D)


def test_configure_optimizers(ddpm_unet2d):
    optim = ddpm_unet2d.configure_optimizers()
    if isinstance(optim, Sequence):
        optim = optim[0]
    elif isinstance(optim, dict):
        optim = optim["optimizer"]

    assert isinstance(
        optim,
        (
            pl_types.Optimizer,
            pl_types.OptimizerConfig,
            pl_types.OptimizerLRSchedulerConfig,
        ),
    )


def test_train():
    import lightning as L
    from ds_utils.huggan_smithsonian_butterflies_subset import (
        DataModule,
        DataLoaderConfig,
        DataModuleConfig,
    )
    from lightning.pytorch.loggers import CSVLogger

    datamodule = DataModule(
        config=DataModuleConfig(
            train_ratio=0.5,
            train_dl_cfg=DataLoaderConfig(batchsize=2, num_workers=8),
            val_dl_cfg=DataLoaderConfig(batchsize=2, num_workers=8),
        )
    )

    model = DDPMUNet2D(
        config=DDPMUNet2DConfig(train=TrainConfig(val_fid_gen_image_n_steps=3))
    )

    trainer = L.Trainer(fast_dev_run=5, logger=[CSVLogger("/")])
    trainer.fit(model, datamodule=datamodule, )
    assert True
