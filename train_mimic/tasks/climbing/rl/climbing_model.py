"""Climbing actor/critic model with separated proprio, history, and ladder fusion."""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import EmpiricalNormalization, HiddenState

from train_mimic.tasks.climbing.rl.ladder_encoders import (
    RELATIVE_RUNG_CONTINUOUS_DIM,
    LadderEncoder,
    build_ladder_encoder,
)
from train_mimic.tasks.tracking.rl.conv1d_encoder import Conv1dEncoder
from train_mimic.tasks.tracking.rl.temporal_cnn_model import _export_input_name


class ClimbingModel(MLPModel):
    """Fuse proprioception, temporal history, and ladder latents before the MLP head."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        cnn_cfg: dict[str, Any] | None = None,
        ladder_cfg: dict[str, Any] | None = None,
    ) -> None:
        if cnn_cfg is None:
            cnn_cfg = {}
        cnn_cfg = dict(cnn_cfg)
        if ladder_cfg is None:
            nested = cnn_cfg.pop("ladder_encoder_cfg", {})
            ladder_cfg = nested if isinstance(nested, dict) else {}
        else:
            cnn_cfg.pop("ladder_encoder_cfg", None)

        ladder_cfg = dict(ladder_cfg)
        self._ladder_observation_mode = str(ladder_cfg.get("mode", "relative_rungs"))

        self._get_obs_dim(obs, obs_groups, obs_set)

        self.history_encoders_dict: dict[str, Conv1dEncoder] = {}
        self.history_latent_dim = 0
        for group_name, obs_dim in zip(self.obs_groups_history, self.obs_dims_history):
            encoder = Conv1dEncoder(input_channels=obs_dim, **cnn_cfg)
            self.history_encoders_dict[group_name] = encoder
            self.history_latent_dim += encoder.output_dim

        self.ladder_encoders_dict: dict[str, LadderEncoder] = {}
        self.ladder_latent_dim = 0
        self._ladder_is_depth: dict[str, bool] = {}
        ladder_output_dim = int(ladder_cfg.get("output_dim", 64))
        ladder_hidden = tuple(ladder_cfg.get("hidden_channels", (32, 64)))
        for group_name, num_rungs, feature_dim, is_depth in zip(
            self.obs_groups_ladder,
            self.obs_ladder_num_rungs,
            self.obs_ladder_feature_dims,
            self.obs_ladder_is_depth,
            strict=True,
        ):
            if is_depth:
                encoder = build_ladder_encoder(
                    mode="depth",
                    num_rungs=0,
                    feature_dim=0,
                    output_dim=ladder_output_dim,
                    hidden_channels=ladder_hidden,
                    kernel_size=int(ladder_cfg.get("kernel_size", 3)),
                    activation=str(ladder_cfg.get("activation", "elu")),
                )
            else:
                encoder = build_ladder_encoder(
                    mode=self._ladder_observation_mode,
                    num_rungs=num_rungs,
                    feature_dim=feature_dim,
                    output_dim=ladder_output_dim,
                    hidden_channels=ladder_hidden,
                    kernel_size=int(ladder_cfg.get("kernel_size", 3)),
                    activation=str(ladder_cfg.get("activation", "elu")),
                )
            self.ladder_encoders_dict[group_name] = encoder
            self._ladder_is_depth[group_name] = is_depth
            self.ladder_latent_dim += encoder.output_dim

        self._obs_normalization_history = obs_normalization
        self._obs_normalization_ladder = obs_normalization

        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
        )

        self.history_encoders = nn.ModuleDict(self.history_encoders_dict)
        self.ladder_encoders = nn.ModuleDict(self.ladder_encoders_dict)

        if obs_normalization:
            history_normalizers: dict[str, nn.Module] = {}
            for group_name, dim in zip(self.obs_groups_history, self.obs_dims_history):
                history_normalizers[group_name] = EmpiricalNormalization(dim)
            self.obs_normalizers_history = nn.ModuleDict(history_normalizers)

            ladder_normalizers: dict[str, nn.Module] = {}
            for group_name, feature_dim, is_depth in zip(
                self.obs_groups_ladder,
                self.obs_ladder_feature_dims,
                self.obs_ladder_is_depth,
                strict=True,
            ):
                if is_depth:
                    ladder_normalizers[group_name] = nn.Identity()
                else:
                    ladder_normalizers[group_name] = EmpiricalNormalization(
                        RELATIVE_RUNG_CONTINUOUS_DIM
                    )
            self.obs_normalizers_ladder = nn.ModuleDict(ladder_normalizers)
        else:
            self.obs_normalizers_history = nn.ModuleDict(
                {g: nn.Identity() for g in self.obs_groups_history}
            )
            self.obs_normalizers_ladder = nn.ModuleDict(
                {g: nn.Identity() for g in self.obs_groups_ladder}
            )

    def _get_obs_dim(
        self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str
    ) -> tuple[list[str], int]:
        active = obs_groups[obs_set]
        obs_dim_1d = 0
        groups_1d: list[str] = []
        groups_history: list[str] = []
        dims_history: list[int] = []
        history_lengths: list[int] = []
        groups_ladder: list[str] = []
        ladder_num_rungs: list[int] = []
        ladder_feature_dims: list[int] = []
        ladder_is_depth: list[bool] = []

        for group_name in active:
            tensor = obs[group_name]
            ndim = len(tensor.shape)
            if ndim == 2:
                groups_1d.append(group_name)
                obs_dim_1d += tensor.shape[-1]
            elif ndim == 3:
                if group_name.endswith("_history"):
                    groups_history.append(group_name)
                    dims_history.append(tensor.shape[-1])
                    history_lengths.append(tensor.shape[1])
                elif group_name.endswith("_ladder"):
                    groups_ladder.append(group_name)
                    ladder_num_rungs.append(tensor.shape[1])
                    ladder_feature_dims.append(tensor.shape[2])
                    ladder_is_depth.append(False)
                else:
                    raise ValueError(
                        f"ClimbingModel 3-D group '{group_name}' must end with "
                        "'_history' or '_ladder'."
                    )
            elif ndim == 4:
                if not group_name.endswith("_ladder"):
                    raise ValueError(
                        f"ClimbingModel 4-D group '{group_name}' must end with '_ladder'."
                    )
                groups_ladder.append(group_name)
                ladder_num_rungs.append(0)
                ladder_feature_dims.append(0)
                ladder_is_depth.append(True)
            else:
                raise ValueError(
                    f"ClimbingModel expects 1-D (B,D), 3-D (B,T,D)/(B,K,F), or "
                    f"4-D (B,1,H,W) ladder obs, got shape {tensor.shape} for "
                    f"'{group_name}'."
                )

        self.obs_groups_history = groups_history
        self.obs_dims_history = dims_history
        self.history_lengths = history_lengths
        self.obs_groups_ladder = groups_ladder
        self.obs_ladder_num_rungs = ladder_num_rungs
        self.obs_ladder_feature_dims = ladder_feature_dims
        self.obs_ladder_is_depth = ladder_is_depth
        self.obs_groups_1d = groups_1d
        return groups_1d, obs_dim_1d

    def _prepare_ladder_obs(self, group_name: str, h: torch.Tensor) -> torch.Tensor:
        if self._ladder_is_depth[group_name]:
            return h
        valid = h[..., 6:7]
        continuous = self.obs_normalizers_ladder[group_name](h[..., :RELATIVE_RUNG_CONTINUOUS_DIM])
        continuous = continuous * valid
        return torch.cat([continuous, valid], dim=-1)

    def _get_latent_dim(self) -> int:
        return self.obs_dim + self.history_latent_dim + self.ladder_latent_dim

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        latent_parts = [super().get_latent(obs, masks, hidden_state)]

        for group_name in self.obs_groups_history:
            h = obs[group_name]
            h = self.obs_normalizers_history[group_name](h)
            h = h.permute(0, 2, 1)
            latent_parts.append(self.history_encoders[group_name](h))

        for group_name in self.obs_groups_ladder:
            h = self._prepare_ladder_obs(group_name, obs[group_name])
            latent_parts.append(self.ladder_encoders[group_name](h))

        return torch.cat(latent_parts, dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        super().update_normalization(obs)
        if self._obs_normalization_history:
            for group_name in self.obs_groups_history:
                h = obs[group_name]
                batch, steps, dim = h.shape
                self.obs_normalizers_history[group_name].update(h.reshape(batch * steps, dim))  # type: ignore[union-attr]
        if self._obs_normalization_ladder:
            for group_name in self.obs_groups_ladder:
                if self._ladder_is_depth[group_name]:
                    continue
                h = obs[group_name]
                valid = h[..., 6:7] > 0.5
                if not bool(valid.any().item()):
                    continue
                continuous = h[..., :RELATIVE_RUNG_CONTINUOUS_DIM]
                flat_valid = valid.squeeze(-1).reshape(-1)
                flat_cont = continuous.reshape(-1, RELATIVE_RUNG_CONTINUOUS_DIM)
                self.obs_normalizers_ladder[group_name].update(flat_cont[flat_valid])  # type: ignore[union-attr]

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        return _OnnxClimbingModel(self, verbose)


class _OnnxClimbingModel(nn.Module):
    def __init__(self, model: ClimbingModel, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.history_normalizers = nn.ModuleList(
            [copy.deepcopy(model.obs_normalizers_history[g]) for g in model.obs_groups_history]
        )
        self.history_encoders = nn.ModuleList(
            [copy.deepcopy(model.history_encoders[g]) for g in model.obs_groups_history]
        )
        self.ladder_normalizers = nn.ModuleList(
            [copy.deepcopy(model.obs_normalizers_ladder[g]) for g in model.obs_groups_ladder]
        )
        self.ladder_encoders = nn.ModuleList(
            [copy.deepcopy(model.ladder_encoders[g]) for g in model.obs_groups_ladder]
        )
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

        self._obs_dim_1d = model.obs_dim
        self._history_input_names = [_export_input_name(g) for g in model.obs_groups_history]
        self._ladder_input_names = [_export_input_name(g) for g in model.obs_groups_ladder]
        self._history_lengths = model.history_lengths
        self._history_dims = model.obs_dims_history
        self._ladder_num_rungs = model.obs_ladder_num_rungs
        self._ladder_feature_dims = model.obs_ladder_feature_dims
        self._ladder_is_depth = list(model.obs_ladder_is_depth)

    def _prepare_ladder_obs(self, idx: int, h: torch.Tensor) -> torch.Tensor:
        if self._ladder_is_depth[idx]:
            return h
        valid = h[..., 6:7]
        continuous = self.ladder_normalizers[idx](h[..., :RELATIVE_RUNG_CONTINUOUS_DIM])
        continuous = continuous * valid
        return torch.cat([continuous, valid], dim=-1)

    def _encode_history(self, obs_history: tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
        if len(obs_history) != len(self.history_encoders):
            raise ValueError(
                f"Expected {len(self.history_encoders)} history inputs, got {len(obs_history)}."
            )
        outputs: list[torch.Tensor] = []
        for h, normalizer, encoder in zip(
            obs_history, self.history_normalizers, self.history_encoders, strict=True
        ):
            x = normalizer(h).permute(0, 2, 1)
            outputs.append(encoder(x))
        return outputs

    def _encode_ladder(self, obs_ladder: tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
        if len(obs_ladder) != len(self.ladder_encoders):
            raise ValueError(
                f"Expected {len(self.ladder_encoders)} ladder inputs, got {len(obs_ladder)}."
            )
        outputs: list[torch.Tensor] = []
        for idx, (h, encoder) in enumerate(zip(obs_ladder, self.ladder_encoders, strict=True)):
            outputs.append(encoder(self._prepare_ladder_obs(idx, h)))
        return outputs

    def forward(
        self,
        obs_1d: torch.Tensor,
        *obs_extra: torch.Tensor,
    ) -> torch.Tensor:
        latent_1d = self.obs_normalizer(obs_1d)
        n_hist = len(self.history_encoders)
        obs_history = obs_extra[:n_hist]
        obs_ladder = obs_extra[n_hist:]
        latent = torch.cat(
            [
                latent_1d,
                *self._encode_history(obs_history),
                *self._encode_ladder(obs_ladder),
            ],
            dim=-1,
        )
        return self.deterministic_output(self.mlp(latent))

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        dummy_1d = torch.zeros(1, self._obs_dim_1d)
        dummy_history = tuple(
            torch.zeros(1, hist_len, dim)
            for hist_len, dim in zip(self._history_lengths, self._history_dims, strict=True)
        )
        dummy_ladder = tuple(
            torch.zeros(1, 1, 64, 64)
            if is_depth
            else torch.zeros(1, num_rungs, feature_dim)
            for is_depth, num_rungs, feature_dim in zip(
                self._ladder_is_depth,
                self._ladder_num_rungs,
                self._ladder_feature_dims,
                strict=True,
            )
        )
        return (dummy_1d, *dummy_history, *dummy_ladder)

    @property
    def input_names(self) -> list[str]:
        return ["obs", *self._history_input_names, *self._ladder_input_names]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
