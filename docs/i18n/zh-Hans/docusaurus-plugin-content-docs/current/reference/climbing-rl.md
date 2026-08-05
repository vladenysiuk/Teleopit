---
sidebar_position: 8
---

# 爬梯 RL 参考

`General-Climbing-G1` **RL 侧**实现：奖励、终止、指标、事件记忆、PPO/模型、课程、训练脚本。

仿真器见 [爬梯仿真器参考](./climbing-simulator.md)。使用说明见 [爬梯教程](../tutorials/climbing.md)。

## 架构

```text
reward/termination/metrics managers ← 读取 task state（非 actor obs）
        ↓
ClimbingModel：Conv1d 历史 + LadderEncoder + MLP(2048…128)
        ↓
ClimbingOnPolicyRunner → rsl_rl PPO（分项 RewardTerm 日志）
```

**固定原则：** 奖励读 `ClimbRewardState`、接触、锁扣、body 张量，**不**读 actor 观测。

## 动作空间

| 项 | 维数 | 说明 |
|----|-----:|------|
| `joint_pos` | 29 | 关节位置偏移 |
| `latch` | 2 | 左右附着/分离 |

## 奖励（默认权重）

| 项 | 权重 | 含义 |
|----|-----:|------|
| `upward_progress` | +10 | 头部（`d435i_link`）ladder 相对高度增量（不用骨盆，避免倒立空翻刷分） |
| `new_higher_attachment` | +2 | 首次附着更高档（一次性） |
| `success` | +20 | 登顶 + 最少附着手数 |
| `time_penalty` | −0.05×dt | 未成功时物理时间惩罚 |
| `action_rate` | −0.01 | 动作变化 L2 |
| `effort` | −1e-4 | 力矩 L2 |
| `invalid_latch` | −0.5 | 无效附着请求 |

`ClimbRewardState` 防止反复 detach/reattach 刷分。

## 终止

成功、坠落（过低/离梯过远）、超时、NaN；锁扣过载默认关闭。

## Episode 指标

| 指标 | 说明 |
|------|------|
| `max_head_height_l` | 头部 / 进度体 ladder 相对高度运行最大值（默认 `d435i_link`） |
| `head_height_l` | 当前头部 / 进度体 ladder 相对高度 |
| `valid_higher_attachments` | 有效更高附着次数 |
| `invalid_latch_count` | 无效附着请求次数 |
| `success` | 成功标志 |
| `hand_contact_fraction` | 手接触均值 |
| `time_to_success` | 首次成功步数，仅对成功回合取平均（环境内哨兵值仍为 −1；无成功的 reset 批次不记录该指标） |
| `torque_saturation_fraction` | 力矩饱和均值 |

## PPO 默认

- `ClimbingModel`，hidden `(2048,1024,512,256,128)`
- `num_steps_per_env=24`，`max_iterations=30000`
- Actor 组：`actor_proprio` + history + `actor_ladder`
- Critic 额外：`critic_privileged`

## 训练预设

| 预设 | 命令 | envs | iters |
|------|------|------|-------|
| Smoke | `--smoke` | 64 | 100 |
| Easy | `--easy` | 256/GPU | 800 |
| Medium | `--medium` | 256/GPU | 800（与 easy 相同 MDP，固定完整 12 级梯子） |

## 多 GPU

```bash
python train_mimic/scripts/train_climb.py --easy --all_gpus --num_envs 128
```

`--num_envs` 为每 GPU 数量；勿传 `--device`。

## 课程

`CurriculumConfig` 的 `randomize_*` 轴；easy / medium 预设全部关闭。建议逐轴开启：初始附着概率 → 间距 → 档数 → 梯子位姿 → 物理 → 初始姿态 → capture 半径。

## 可学习性

`debug_climb.py --mode learnability` 对比 zero/random 与短训 checkpoint 的进度高度（默认头部 / `d435i_link`）与有效附着次数。

## 回放

`play_climb.py --easy` / `--medium` 须与对应训练预设 MDP 一致。支持 native viewer、Viser（SSH）与无头 `--video`（EGL）。视频模式默认将多次 reseed 拼成一条 Full HD mp4，并以 30% 速度播放（`--video-clips 8`，`--video-speed 0.3`，`1920x1080`）。

## 测试

```bash
python scripts/dev/run_climbing_regression.py
pytest tests/test_climbing_stage5.py tests/test_climbing_stage8.py tests/test_climbing_stage9.py -v
```

## 已知 RL 限制

- 尚无爬梯 ONNX 导出合约
- 早期 `invalid_latch` 可能主导；easy 预设保证初始附着
- 尚无 broad 域随机化

Stage 笔记：`STAGE5/8/9/10*.md`。
