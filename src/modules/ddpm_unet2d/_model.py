import functools
import json
import os
from enum import StrEnum
from typing import (
    Literal,
    List,
    Self,
    Any,
    Callable,
    TypedDict,
    Iterable,
    Sequence,
)

import torch
import torch.nn as nn
from datasets import tqdm
from diffusers import UNet2DModel, DDPMScheduler
from lightning import LightningModule
from lightning.pytorch.utilities.types import (
    STEP_OUTPUT,
    OptimizerLRScheduler,
    OptimizerLRSchedulerConfig,
    LRSchedulerConfigType,
)
from loguru import logger
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    model_validator,
    field_serializer,
)
from torch.optim.optimizer import ParamsT
from torchmetrics.image.fid import FrechetInceptionDistance

UNet2DDownBLock = Literal[
    "DownBlock2D",
    "ResnetDownsampleBlock2D",
    "AttnDownBlock2D",
    "CrossAttnDownBlock2D",
    "SimpleCrossAttnDownBlock2D",
    "SkipDownBlock2D",
    "AttnSkipDownBlock2D",
    "DownEncoderBlock2D",
    "AttnDownEncoderBlock2D",
    "KDownBlock2D",
    "KCrossAttnDownBlock2D",
]

Unet2DMidBlock = Literal[
    "UNetMidBlock2DCrossAttn", "UNetMidBlock2DSimpleCrossAttn", "UNetMidBlock2D"
]

UNet2DUpBlock = Literal[
    "UpBlock2D",
    "ResnetUpsampleBlock2D",
    "AttnUpBlock2D",
    "CrossAttnUpBlock2D",
    "SimpleCrossAttnUpBlock2D",
    "SkipUpBlock2D",
    "AttnSkipUpBlock2D",
    "UpEncoderBlock2D",
    "AttnUpEncoderBlock2D",
    "KUpBlock2D",
    "KCrossAttnUpBlock2D",
]


class UNet2DConfig(BaseModel):
    # Image shape i/o
    sample_size: int | tuple[int, int] | None = None
    """Dimensions must be a multiple of `2 ** (len(block_out_channels) -1)`."""

    in_channels: int = 3
    out_channels: int = 3

    # Unet blocks
    down_block_types: List[UNet2DDownBLock] = Field(
        default_factory=lambda _: [
            *["DownBlock2D" for _ in range(4)],
            "AttnDownBlock2D",
            "DownBlock2D",
        ]
    )
    mid_block_type: Unet2DMidBlock | None = "UNetMidBlock2D"
    up_block_types: List[UNet2DUpBlock] = Field(
        default_factory=lambda _: [
            "UpBlock2D",
            "AttnUpBlock2D",
            *["UpBlock2D" for _ in range(4)],
        ]
    )
    block_out_channels: List[PositiveInt] = Field(
        default_factory=lambda _: [128, 128, 256, 256, 512, 512]
    )

    # Misc
    act_fn: str = "silu"
    time_embedding_type: str = "positional"

    center_input_sample: bool = False
    """Scale [0,1] to [-1,1] or not by apply 'sample = 2 * sample - 1.0' in forward method."""

    layers_per_block: int = 2
    """
    for i in range(layers_per_block):
       in_channels = in_channels if i == 0 else out_channels
       resnets.append(ResnetBlock2D( ...
    """

    # Rarely change, See more in 'UNet2DModel' constructor.
    time_embedding_dim: int | None = None
    freq_shift: int = 0
    flip_sin_to_cos: bool = True
    mid_block_scale_factor: float = 1
    downsample_padding: int = 1
    downsample_type: str = "conv"
    upsample_type: str = "conv"
    dropout: float = 0.0
    attention_head_dim: int | None = 8
    norm_num_groups: int = 32
    attn_norm_num_groups: int | None = None
    norm_eps: float = 1e-5
    resnet_time_scale_shift: str = "default"
    add_attention: bool = True
    class_embed_type: str | None = None
    num_class_embeds: int | None = None
    num_train_timesteps: int | None = None

    _sample_scale_factor: PositiveInt = None

    @model_validator(mode="after")
    def _validate(self) -> Self:
        assert (
            (l := len(self.block_out_channels))
            == len(self.up_block_types)
            == len(self.down_block_types)
        )

        self._sample_scale_factor = 2 ** (l - 1)

        if self.sample_size is not None:
            if not isinstance(self.sample_size, Sequence):
                assert isinstance(self.sample_size, int)
                self.sample_size = (self.sample_size, self.sample_size)
            for s in self.sample_size:
                assert s % self._sample_scale_factor == 0

        return self

    @classmethod
    def from_unet(cls, unet: UNet2DModel):
        assert isinstance(unet, UNet2DModel)
        return cls.model_validate(unet.config)


class PredictionType(StrEnum):
    EPSILON = "epsilon"
    SAMPLE = "sample"
    V_PREDICTION = "v_prediction"


class SchedulerConfig(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_default=True, use_enum_values=True
    )
    num_train_timesteps: int = 1000

    # Noise scheduler
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: Literal["linear", "caled_linear", "squaredcos_cap_v2", "sigmoid"] = (
        "linear"
    )
    trained_betas: List[float] | None = None

    prediction_type: PredictionType = PredictionType.EPSILON
    """The type that Unet/Model output assumed of. See Section 4 in https://arxiv.org/abs/2202.00512"""

    # Rarely change, see details in `DDPMScheduler` constructor.
    variance_type: Literal[
        "fixed_small",
        "fixed_small_log",
        "fixed_large",
        "fixed_large_log",
        "learned",
        "learned_range",
    ] = "fixed_small"
    clip_sample: bool = True
    thresholding: bool = False
    dynamic_thresholding_ratio: float = 0.995
    clip_sample_range: float = 1.0
    sample_max_value: float = 1.0
    timestep_spacing: Literal["leading"] = "leading"
    steps_offset: int = 0
    rescale_betas_zero_snr: bool = False

    @classmethod
    def from_scheduler(cls, scheduler: DDPMScheduler) -> Self:
        assert isinstance(scheduler, DDPMScheduler)
        return cls.model_validate(scheduler.config)


class Metric(StrEnum):
    TRAIN_LOSS = "train_loss"
    VAL_LOSS = "val_loss"
    VAL_FID = "val_fid"


# Note, this is declared instead of lambda for pickable
def _loss_fn_default_factory():
    return nn.MSELoss()


def _default_optimizer_factory(p: ParamsT, lr=1e-4, **kwargs) -> OptimizerLRScheduler:
    optimizer = torch.optim.AdamW(p, lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.1,
        patience=5,
    )
    return OptimizerLRSchedulerConfig(
        optimizer=optimizer,
        lr_scheduler=LRSchedulerConfigType(
            scheduler=lr_scheduler,
            name="default_optimizer",
            # Updating frequency of the scheduler ( call scheduler.step())
            interval="epoch",
            frequency=1,
            # The value for scheduler to depend on to update
            monitor=Metric.TRAIN_LOSS,
            strict=True,  # Must have monitor value
        ),
    )


class TrainConfig(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_default=True,
        use_enum_values=True,
        validate_assignment=True,
    )
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = Field(
        default_factory=_loss_fn_default_factory
    )
    """Function that receive two equal-shape FloatTensors in order (prediction, target), return FloatTensors with len(shape)==0."""

    optimizers_factory: Callable[[ParamsT, ...], OptimizerLRScheduler] = (
        _default_optimizer_factory
    )

    metrics: List[Metric] = ["train_loss", "val_loss", "val_fid"]
    """Metrics to log"""

    val_fid_gen_image_n_steps: PositiveInt | None = Field(None)
    """This field only used if `val_fid` is set in `metrics`.
    If None, it will be `0.1*num_train_timesteps`.
    If > `num_train_timesteps`, it's equal to `num_train_timesteps`.
    """

    @field_serializer("loss_fn")
    def _serialize_loss_fn(self, loss_fn):
        return repr(loss_fn)

    @field_serializer("optimizers_factory")
    def _serialize_optimizer_factory(self, of):
        optim = of([torch.tensor(0)])
        return f"(Iterable[torch.Parameters]) -> ({optim})"

    @model_validator(mode="after")
    def _validate(self) -> Self:
        assert (
            len(
                self.loss_fn(
                    torch.ones([2, 3], dtype=torch.float16),
                    torch.zeros([2, 3], dtype=torch.float16),
                ).shape
            )
            == 0
        )
        return self


class DDPMUNet2DConfig(BaseModel):
    model_config = ConfigDict(validate_default=True, validate_assignment=True)

    unet: UNet2DConfig = Field(default_factory=UNet2DConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if self.train.val_fid_gen_image_n_steps:
            self.train.val_fid_gen_image_n_steps = min(
                self.train.val_fid_gen_image_n_steps, self.scheduler.num_train_timesteps
            )
        else:
            self.train.val_fid_gen_image_n_steps = int(
                0.1 * self.scheduler.num_train_timesteps
            )

        return self


class StepOutput(TypedDict):
    loss: torch.FloatTensor


_call_count: int = 0
try:
    LOG_FUNC: int = int(os.getenv("LOG_FUNC", 0))
except:
    LOG_FUNC: int = 0


def log_func(func):
    if not LOG_FUNC:
        return func

    global _call_count

    @functools.wraps(func)
    def _func(*args, **kwargs):
        global _call_count
        logger.debug(
            f"{func.__name__}:{_call_count:,}"
            # f"args: {args},\n"
            # f"kwargs: {kwargs}\n"
        )
        _call_count += 1
        return func(*args, **kwargs)

    return _func


# noinspection PyMethodMayBeStatic,PyTypeChecker
class DDPMUNet2D(LightningModule):
    """
    The input images, noises must be centralized, such as map from standard [0,255] to [-1,1].
     Or noises random sampled from N(0,1).
    """

    def __init__(
        self,
        config: DDPMUNet2DConfig = None,
        *,
        unet: UNet2DModel | None = None,
        scheduler: DDPMScheduler | None = None,
    ):
        """

        Args:
            config: If None, It will create default config, see `DDPMUNet2DConfig` class for details.
            unet: If given, the `unet` from `config` will be overridden
            scheduler: If give, the `scheduler` from `config` will be overridden.
        """
        super().__init__()
        config = config or DDPMUNet2DConfig()  # Default
        assert isinstance(config, DDPMUNet2DConfig)

        # Overridden
        if unet:
            config.unet = UNet2DConfig.from_unet(unet)
            self.unet = unet
        else:
            self.unet = UNet2DModel.from_config(config.unet.model_dump())

        if scheduler:
            config.scheduler = SchedulerConfig.from_scheduler(scheduler)
            self.scheduler = scheduler
        else:
            self.scheduler = DDPMScheduler.from_config(config.scheduler.model_dump())

        self.save_hyperparameters({"config": config})

        for metric in self.metrics:
            if metric == Metric.TRAIN_LOSS:
                self.train_loss: torch.FloatTensor | None = None
            elif metric == Metric.VAL_LOSS:
                self.val_loss: torch.FloatTensor | None = None
            else:
                assert metric == Metric.VAL_FID
                self.all_real_loaded: bool = False
                """Should checking this flag when call update to real images. This flag is set at the first end validation epoch."""
                self.fid_metric = FrechetInceptionDistance(
                    64,
                    reset_real_features=False,  # Val dataset not change. So does not need to reset this.
                    normalize=False,  # Images must be uint8 Tensor.
                )
                self.val_fid_gen_image_n_steps: int = (
                    self.config.train.val_fid_gen_image_n_steps
                )
        logger.debug(
            f"Init DDPMUnet2D with config:\n" f"{self.config.model_dump_json(indent=4)}"
        )

    @property
    def config(self) -> DDPMUNet2DConfig:
        return self.hparams.config

    @property
    def metrics(self) -> list[Metric | str]:
        return self.config.train.metrics

    @property
    def loss_fn(
        self,
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor | torch.FloatTensor]:
        """
        Returns:
            Function that receive two equal-shape FloatTensors in order (prediction, target), return FloatTensors with len(shape)==0
        """
        return self.config.train.loss_fn

    @log_func
    def log_metrics(self):
        metrics = {}
        for metric in self.metrics:
            if metric == Metric.TRAIN_LOSS:
                assert isinstance(self.train_loss, torch.Tensor)
                metrics[metric] = self.train_loss.mean().item()
                self.train_loss = None
            elif metric == Metric.VAL_LOSS:
                assert isinstance(self.val_loss, torch.Tensor)
                metrics[metric] = self.val_loss.mean().item()
                self.val_loss = None
            else:
                assert metric == Metric.VAL_FID
                metrics[metric] = self.fid_metric.compute().item()
                self.fid_metric.reset()
        logger.debug(json.dumps(metrics, indent=4))
        self.log_dict(metrics, logger=True, sync_dist=True)

    @log_func
    def update_metrics(
        self,
        metric: Metric,
        value: torch.FloatTensor | torch.ByteTensor | torch.Tensor,
        *,
        real: bool = True,
    ):
        assert metric in self.metrics
        assert torch.is_tensor(value)

        if metric == Metric.TRAIN_LOSS:
            assert torch.is_floating_point(value)
            assert value.size() == torch.Size([])
            self.train_loss = (
                torch.cat([self.train_loss, value[None]])
                if self.train_loss is not None
                else value[None]
            )
        elif metric == Metric.VAL_LOSS:
            assert torch.is_floating_point(value)
            assert value.size() == torch.Size([])
            self.val_loss = (
                torch.cat([self.val_loss, value[None]])
                if self.val_loss is not None
                else value[None]
            )
        else:
            assert metric == Metric.VAL_FID
            if real and self.all_real_loaded:
                raise ValueError(
                    "`all_real_loaded` flag is set to True, so cannot update more real features/images."
                )
            # assert (mx := value.max()) <= 1.0 or torch.isclose(mx, torch.tensor(1.0))
            # assert (mn := value.min()) >= 0 or torch.isclose(mn, torch.tensor(0.0))
            assert value.dtype == torch.uint8
            self.fid_metric.update(value, real)

    def validate_input_samples(self, samples: torch.Tensor, **kwargs) -> torch.Tensor:
        """

        Args:
            samples: FloatTensor with shape (C,H,W) or (N,C,H,W)
            kwargs: boolean option values with keys (all False by default):
                - `validated` : If True, means already validated, so don't do anything, just return input samples.
                - `center`: If True, assert input ranges in [-1,1]
                - `size` or `shape`: If True, check for sanity with Unet sample_size.
             If there's one key is True, do unsqueeze, check for channel and floating point.
        Returns:
            Tensor

        """
        # Images must be FloatTensor and values in range [-1,1]
        if not kwargs.get("validated", False):

            # Common validations
            if len(samples.shape) == 3:
                samples = samples.unsqueeze(0)
            assert len(s := samples.shape) == 4
            assert s[1] == 3
            assert torch.is_tensor(samples) and torch.is_floating_point(samples)
            samples = samples.to(device=self.device, dtype=self.dtype)

            # Center
            if kwargs.get("center", False):
                assert (mx := samples.max()) <= 1.0 or torch.isclose(
                    mx, torch.tensor(1.0)
                )
                assert (mn := samples.min()) >= -1.0 or torch.isclose(
                    mn, torch.tensor(-1.0)
                )

            if kwargs.get("size", False) or kwargs.get("shape", False):
                if (ss := self.config.unet.sample_size) is not None:
                    if not s[2:] == ss:
                        raise ValueError(
                            f"Samples size (HxW) must be equal to Unet.sample_size. Got {s[2:]} and {ss}."
                        )
                else:
                    for s in s[2:]:
                        if not (s % self.config.unet._sample_scale_factor) == 0:
                            raise ValueError(
                                f"Input size must be the scaling of 2**len(block_out_channels)=={self.config.unet._sample_scale_factor} with integer. Got {ss}"
                            )

        return samples

    @log_func
    def forward(
        self, noises: torch.Tensor, timesteps: torch.Tensor | float | int, **kwargs: Any
    ) -> torch.Tensor:
        """
        Validate input and call UNet forward.
        Args:
            noises:The noisy images input FloatTensor with the following shape `(N, C, H, W)` or `(C,H,W)`.
            timesteps: 0-D or 1-D tensor or float or int. This will be converted to 1-D Tensor later if given 0-D.
            **kwargs:

        Returns:
            A Tensor which with the same shape`(N,C,H,W)` as `noises`, N=1 if noises shape is `(C,H,W)`  arg.
             It can be used as `sample, v_prediction or epsilon`, see `self.config.scheduler.prediction_type`
        """

        noises = self.validate_input_samples(noises, size=True, **kwargs)

        if torch.is_tensor(timesteps):
            # 1D-Tensor
            if len(ts := timesteps.shape) == 1:
                assert (
                    ts[0] == noises.shape[0]
                )  # Will be used for `timesteps = timesteps * torch.ones(sample.shape[0])`
            # 0-D Tensor
            else:
                assert len(ts) == 0
        model_output = self.unet(noises, timesteps).sample
        assert model_output.shape == noises.shape

        return model_output

    @log_func
    def generate(
        self,
        noise: torch.Tensor,
        num_steps_or_timesteps: int | Sequence[int] = None,
        *,
        clamp_output: Literal["center", True, False] = True,
        **kwargs,
    ) -> torch.Tensor:
        """

        Args:
            noise: A FloatTensor noisy batch shape `(C,H,W)` or `(N,C,H,W)`, should be sampled by N(0,1) distribution.
            num_steps_or_timesteps: If None, use `num_train_timesteps`.

            clamp_output:
             - If True, clamp and scale output value to range [0,1] ( Default).
             - If "center", clamp output value to range [-1,1]
             - If False, return the raw sample (output of model) directly (tend to centralize to 0.0).


        Returns:
            An FloatTensor shape `(N,C,H,W)`.
        """
        noise = self.validate_input_samples(noise, size=True, **kwargs)

        num_steps_or_timesteps = (
            num_steps_or_timesteps or self.config.scheduler.num_train_timesteps
        )
        if isinstance(num_steps_or_timesteps, Sequence):
            self.scheduler.set_timesteps(
                device=self.device, timesteps=num_steps_or_timesteps
            )
        else:
            self.scheduler.set_timesteps(num_steps_or_timesteps, device=self.device)

        sample = noise
        with torch.inference_mode():
            for step in tqdm(
                self.scheduler.timesteps,
                bar_format=f"Generating {noise.size(0)} samples - Steps: {{n_fmt}}/{{total_fmt}} [{{bar}}] {{rate_fmt}}",
            ):
                model_output = self(sample, step, validated=True)
                sample = self.scheduler.step(model_output, step, sample).prev_sample
        if clamp_output:
            if clamp_output == "center":
                return sample.clamp(-1, 1)
            return (sample * 0.5 + 0.5).clamp(0, 1)
        return sample

    def create_timesteps(self, size: Sequence[int]) -> torch.IntTensor:
        timesteps: torch.IntTensor = torch.randint(
            0,
            self.config.scheduler.num_train_timesteps,
            size,
            device=self.device,
            dtype=torch.int32,
        )
        return timesteps

    def create_noises(self, size: Sequence[int]):
        return torch.randn(size, dtype=self.dtype, device=self.device)

    @log_func
    def calculate_loss(self, images: torch.Tensor, **kwargs) -> torch.FloatTensor:

        images = self.validate_input_samples(images, size=True, center=True, **kwargs)

        # Randomize timesteps, noises
        timesteps = self.create_timesteps((images.size(0),))
        noises = self.create_noises(images.size())

        # Add noises to images
        noisy_images = self.scheduler.add_noise(images, noises, timesteps)

        # Forward and calculate loss
        prediction = self(noisy_images, timesteps, validated=True)

        # Create target
        if pred_type := self.config.scheduler.prediction_type == PredictionType.EPSILON:
            target = noises
        elif pred_type == PredictionType.SAMPLE:
            target = images
        elif pred_type == PredictionType.V_PREDICTION:
            target = self.scheduler.get_velocity(images, noises, timesteps)
        else:
            raise RuntimeError(f"The `prediction_type`=`{pred_type}` is not supported.")
        return self.loss_fn(prediction, target)

    @log_func
    def training_step(
        self, images: torch.Tensor, *args: Any, **kwargs: Any
    ) -> STEP_OUTPUT:
        return StepOutput(loss=self.calculate_loss(images))

    @log_func
    def on_train_batch_end(
        self, outputs: StepOutput, batch: Any, batch_idx: int
    ) -> None:
        if (m := Metric.TRAIN_LOSS) in self.metrics:
            self.update_metrics(m, outputs["loss"])

    @log_func
    def on_train_epoch_end(self) -> None:
        self.log_metrics()
        optims = self.optimizers()
        if not isinstance(optims, Sequence):
            optims = [optims]
        lrs = {}
        for i, optim in enumerate(optims):
            for j, pg in enumerate(optim.optimizer.param_groups):
                lrs[f"optim_{i}-pg_{j}_lr"] = pg["lr"]
        logger.debug(json.dumps(lrs, indent=4))
        self.log_dict(lrs, logger=True, sync_dist=True)

    @log_func
    def validation_step(
        self, images: torch.Tensor, *args: Any, **kwargs: Any
    ) -> STEP_OUTPUT:
        images = self.validate_input_samples(
            images,
            center=True,
            size=True,
        )

        for metric in self.metrics:
            if metric == Metric.VAL_LOSS:
                self.update_metrics(metric, self.calculate_loss(images, validated=True))
            if metric == Metric.VAL_FID:
                noises = self.create_noises(images.size())
                # Update generated images
                gen_images = self.generate(
                    noises,
                    self.val_fid_gen_image_n_steps,
                    clamp_output=True,
                    validated=True,
                )
                self.update_metrics(metric, (gen_images * 255).byte(), real=False)
                if not self.all_real_loaded:
                    self.update_metrics(
                        metric, ((images * 0.5 + 0.5) * 255).byte(), real=True
                    )

    @log_func
    def on_validation_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """In the hook loop, this method call directly after `validation_step`, so
        just implement every thing in `validation_step` method.
        See Also: https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#hooks
        """
        return None

    @log_func
    def on_validation_epoch_end(self) -> None:
        self.all_real_loaded = True

    @log_func
    def configure_optimizers(self) -> OptimizerLRScheduler:
        params: Iterable[torch.Tensor] = self.parameters()
        return self.config.train.optimizers_factory(params)


# model = DDPMUNet2D()
# print(model.config.model_dump_json(indent=4))
# print(
#     model.config.scheduler.prediction_type, type(model.config.scheduler.prediction_type)
# )
