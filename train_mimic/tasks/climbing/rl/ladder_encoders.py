"""Ladder perception encoders for General-Climbing-G1."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from train_mimic.tasks.tracking.rl.conv1d_encoder import Conv1dEncoder


# Continuous geometry dims per rung slot; valid bit is excluded from normalization.
RELATIVE_RUNG_CONTINUOUS_DIM = 6


class LadderEncoder(ABC):
    """Common output width for vector and depth ladder encoders."""

    @property
    @abstractmethod
    def output_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class RelativeRungVectorEncoder(nn.Module, LadderEncoder):
    """Encode ``[B, K, F]`` relative-rung vectors into a fixed latent."""

    def __init__(
        self,
        *,
        num_rungs: int,
        feature_dim: int,
        output_dim: int = 64,
        hidden_channels: tuple[int, ...] = (32, 64),
        kernel_size: int = 3,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        self._num_rungs = num_rungs
        self._feature_dim = feature_dim
        self._encoder = Conv1dEncoder(
            input_channels=feature_dim,
            output_channels=hidden_channels,
            kernel_size=kernel_size,
            activation=activation,
            global_pool="avg",
        )
        if self._encoder.output_dim != output_dim:
            self._proj = nn.Linear(self._encoder.output_dim, output_dim)
        else:
            self._proj = nn.Identity()
        self._output_dim = output_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``[B, K, F]`` relative-rung tensor with valid bit in the last channel.

        Returns:
            ``[B, output_dim]`` latent vector.
        """
        if x.ndim != 3:
            raise ValueError(f"Expected [B, K, F] ladder input, got shape {tuple(x.shape)}.")
        if x.shape[1] != self._num_rungs or x.shape[2] != self._feature_dim:
            raise ValueError(
                "RelativeRungVectorEncoder shape mismatch: "
                f"expected (*, {self._num_rungs}, {self._feature_dim}), got {tuple(x.shape)}."
            )
        valid = x[..., 6:7]
        masked = x.clone()
        masked[..., :RELATIVE_RUNG_CONTINUOUS_DIM] = (
            masked[..., :RELATIVE_RUNG_CONTINUOUS_DIM] * valid
        )
        h = masked.permute(0, 2, 1)  # (B, F, K)
        return self._proj(self._encoder(h))


class DepthLadderEncoder(nn.Module, LadderEncoder):
    """Shape-level depth encoder interface for future ``DepthCameraProvider``."""

    def __init__(
        self,
        *,
        output_dim: int = 64,
        hidden_channels: tuple[int, ...] = (16, 32, 64),
        activation: str = "elu",
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = 1
        act_cls = nn.ELU if activation == "elu" else nn.ReLU
        for out_ch in hidden_channels:
            layers.extend(
                [
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                    act_cls(),
                ]
            )
            in_ch = out_ch
        layers.extend([nn.AdaptiveAvgPool2d(1), nn.Flatten(start_dim=1)])
        self._cnn = nn.Sequential(*layers)
        self._proj = nn.Linear(hidden_channels[-1], output_dim)
        self._output_dim = output_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``[B, 1, H, W]`` depth image batch.

        Returns:
            ``[B, output_dim]`` latent vector.
        """
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"Expected [B, 1, H, W] depth input, got shape {tuple(x.shape)}.")
        return self._proj(self._cnn(x))


def build_ladder_encoder(
    *,
    mode: str,
    num_rungs: int,
    feature_dim: int,
    output_dim: int,
    hidden_channels: tuple[int, ...],
    kernel_size: int,
    activation: str,
) -> LadderEncoder:
    """Construct the ladder encoder selected by ``ladder_observation.mode``."""
    if mode == "depth":
        return DepthLadderEncoder(
            output_dim=output_dim,
            hidden_channels=hidden_channels,
            activation=activation,
        )
    if mode == "relative_rungs":
        return RelativeRungVectorEncoder(
            num_rungs=num_rungs,
            feature_dim=feature_dim,
            output_dim=output_dim,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            activation=activation,
        )
    raise ValueError(
        f"Unsupported ladder_observation.mode={mode!r}. Expected 'relative_rungs' or 'depth'."
    )


class DepthCameraProvider(ABC):
    """Future depth perception insertion point; not implemented in Stage 4."""

    @abstractmethod
    def capture(self) -> torch.Tensor:
        raise NotImplementedError


class NotImplementedDepthCameraProvider(DepthCameraProvider):
    """Explicit failure when a depth provider is requested but unavailable."""

    def capture(self) -> torch.Tensor:
        raise NotImplementedError(
            "DepthCameraProvider is not implemented for General-Climbing-G1. "
            "Use RelativeRungProvider for the initial privileged ladder path."
        )
