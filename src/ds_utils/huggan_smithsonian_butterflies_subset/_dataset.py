from collections.abc import Iterable
from math import ceil, isclose
from typing import TypedDict, Sequence, Callable, Literal, Self

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from datasets import load_dataset, Dataset as ArrDataset
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data.dataset import _T_co
from torchvision.transforms import v2


class SampleType(TypedDict):
    raw_images: list[Image.Image] | Image.Image
    images: torch.Tensor
    """
    Float tensor with shape (batchsize, channels, height, width) or (C,H,W), values must in range [-1,1]
    """


def check_sample(samples: SampleType):
    imgs = samples["images"]
    assert imgs.max() <= 1.0 or isclose(imgs.max(), 1.0, abs_tol=1e-6)
    assert imgs.min() >= -1.0 or isclose(imgs.min(), -1.0, abs_tol=1e-6)
    assert torch.is_floating_point(imgs)

    if len(imgs.shape) == 4:
        imgs = imgs[0]
    assert len(s := imgs.shape) == 3
    assert s[0] == 3


class DefaultTransformKwargs(TypedDict):
    resize: int | tuple[int, int]
    """The size (H,W) that all images must be for batching."""

    p_h_flip: float


class HugganSmithsonianButterfliesSubsetDataset(TorchDataset[SampleType]):
    """
    Examples:

        from ds_utils.huggan_smithsonian_butterflies_subset import HugganSmithsonianButterfliesSubsetDataset

        dataset = HugganSmithsonianButterfliesSubsetDataset()

        print(dataset)

        dataset[:5]["images"].shape # torch.Size([5, 3, 128, 128])

        dataset.show_images(range(5))

        dataset.show_images(range(5),raw=False)

    """

    ds_path = "huggan/smithsonian_butterflies_subset"
    df_tf_kwargs = DefaultTransformKwargs(resize=(128, 128), p_h_flip=0.5)

    def __init__(
        self,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        mode: Literal["train", "val"] = "train",
        **default_tf_kwargs,
    ):
        """

        Args:
            transform: Function with input and output are  Float tensor with shape (batchsize, channels, height, width) or (C,H,W)
             values must in range [-1,1].
            **default_tf_kwargs:
        """

        self.tf_kw: DefaultTransformKwargs = self.df_tf_kwargs
        self.tf_kw.update(default_tf_kwargs)

        self._preprocess_images = v2.Compose(
            [
                v2.ToImage(),  # Always resize in the very first of pipeline, when data is in uint8
                v2.Resize(self.tf_kw["resize"]),
                v2.ToDtype(dtype=torch.float, scale=True),
                v2.Normalize([0.5], [0.5]),
            ]
        )

        if not transform:
            transform = v2.Compose(
                [
                    v2.RandomHorizontalFlip(
                        self.tf_kw["p_h_flip"]
                    ),  # convert to [-1,1]
                ]
            )
        self._transform: Callable[[torch.Tensor], torch.Tensor] = transform

        self.mode = mode
        ds = default_tf_kwargs.pop("__dataset__", None)  # This is set by split method
        self._dataset: ArrDataset = ds or self.load_dataset()
        self._dataset.set_transform(lambda ex: self.transform(ex["image"]))

    @classmethod
    def load_dataset(cls):
        return load_dataset(cls.ds_path, split="train")

    @classmethod
    def split(
        cls,
        train_ratio: float = 0.8,
        shuffle: bool = True,
        *dataset_args,
        **dataset_kwargs,
    ) -> tuple[Self, Self]:
        assert 1 > train_ratio > 0
        assert "__dataset__" not in dataset_kwargs
        ds_dict = cls.load_dataset().train_test_split(
            train_size=train_ratio, shuffle=shuffle
        )

        return (
            cls(
                *dataset_args,
                **dataset_kwargs,
                mode="train",
                __dataset__=ds_dict["train"],
            ),
            cls(
                *dataset_args, **dataset_kwargs, mode="val", __dataset__=ds_dict["test"]
            ),
        )

    def transform(
        self, images: list[Image.Image] | np.ndarray | torch.Tensor
    ) -> SampleType:
        """
        Transform batch of images.
        Args:
            images: batch of images

        Returns:
            SampleType

        """
        processed_images = torch.stack(self._preprocess_images(images), 0)
        tf_images = (
            self._transform(processed_images)
            if self.mode == "train"
            else processed_images
        )

        samples = SampleType(images=tf_images, raw_images=images)
        check_sample(samples)
        return samples

    def __getitem__(self, index) -> _T_co:
        return self._dataset[index]

    def __len__(self):
        return len(self._dataset)

    def __repr__(self):
        return self._dataset.__repr__()

    def show_images(
        self,
        indexes: Sequence[int] | range | Iterable,
        img_per_row=None,
        raw: bool = True,
    ):

        images: list[Image.Image] | np.ndarray | torch.Tensor

        # noinspection PyTypeChecker
        samples: SampleType = self[indexes]
        if raw:
            images = samples["raw_images"]
        else:
            images = (samples["images"] / 2 + 0.5).permute(0, 2, 3, 1).numpy()
        self.show(images, img_per_row)

    @classmethod
    def show(
        cls,
        images: np.ndarray | list[Image.Image],
        img_per_row=None,
        **kwargs
    ):
        n_imgs = len(images)
        img_per_row = img_per_row or max(5, int(n_imgs / 2))
        n_rows = ceil(n_imgs / img_per_row)
        fig, axes = plt.subplots(
            n_rows, img_per_row, figsize=(img_per_row * 2, n_rows * 2)
        )
        axes = axes.flatten()

        for i, img in enumerate(images):
            if isinstance(img, Image.Image):
                title = f"Size (WxH): {img.size}"
            else:
                assert isinstance(img, np.ndarray)
                title = f"Size (HxW): {img.shape[:-1]}"

            ax = axes[i]
            ax.imshow(img)
            ax.set_title(title)
            ax.axis("off")  # Hide axes ticks and labels

        for j in range(len(images), len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()


Dataset = HugganSmithsonianButterfliesSubsetDataset

if __name__ == "__main__":
    dataset = Dataset()
    print(dataset)
    dataset.show_images(range(5))
    dataset.show_images(range(5), raw=False)
