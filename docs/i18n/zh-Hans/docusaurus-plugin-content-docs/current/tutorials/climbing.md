---
sidebar_position: 6
---

# G1 爬梯训练（Climbing）

在 MuJoCo 中使用 `General-Climbing-G1` 任务训练 Unitree G1 爬梯策略。该任务与 `General-Tracking-G1` **相互独立**，**不**共享 167D 追踪 ONNX 合约。

:::info
仿真器实现细节见 [爬梯仿真器参考](../reference/climbing-simulator.md)。奖励、PPO 与课程见 [爬梯 RL 参考](../reference/climbing-rl.md)。
:::

## 前置条件

```bash
pip install -e ".[train]"
python scripts/setup/download_assets.py --only robots
python -c "import train_mimic.tasks; print('ok')"
```

Canonical 机器人模型：`assets/robots/unitree_g1/g1_29dof.xml`（与追踪任务共用）。

| 常量 | 值 |
|------|-----|
| 任务 ID | `General-Climbing-G1` |
| 实验名 | `g1_general_climbing` |
| 物理频率 | 200 Hz（`sim_dt=0.005`） |
| 策略频率 | 50 Hz（`decimation=4`） |

## 快速验证流程

长时间训练前建议按顺序执行：

```bash
python scripts/dev/run_climbing_regression.py --quick
mjpython train_mimic/scripts/debug_climb.py --mode scene --seed 42
python train_mimic/scripts/train_climb.py --smoke
python train_mimic/scripts/train_climb.py --easy --num_envs 64 --max_iterations 50
```

## 调试模式（`debug_climb.py`）

macOS 上 viewer 模式请使用 `mjpython`。

| 模式 | 用途 | Viewer |
|------|------|--------|
| `scene` | 采样梯子拓扑 | 是 |
| `contacts` | 手/横档接触识别 | 是 |
| `latch` | 附着/ detach 脚本 | 是 |
| `observations` | 相对横档观测 | 是 |
| `rewards` | 分项奖励 | 是 |
| `hold-pose` | 四接触静态持握 | 是 |
| `zero` / `random` | MDP  rollout | 无头 |
| `ppo-smoke` | PPO 基础设施 | 无头 |
| `baseline-compare` / `learnability` | 可学习性对比 | 无头 |

## 训练

### 冒烟测试（Stage 8）

```bash
python train_mimic/scripts/train_climb.py --smoke
```

### Easy 可学习性预设（Stage 9）

固定 6 级垂直梯子、辅助 hold 复位、双手初始附着：

```bash
python train_mimic/scripts/train_climb.py --easy
```

### 多 GPU（单节点）

`--num_envs` 为 **每块 GPU** 的环境数。

```bash
python train_mimic/scripts/train_climb.py --easy --all_gpus --num_envs 128
python train_mimic/scripts/train_climb.py --easy --gpu_ids 0 1 2 3 --num_envs 128
```

注意：多 GPU 模式下不要传 `--device`。4× A100（80GB）上 `--easy` 建议从 **128 envs/GPU** 起步。

### 断点续训

```bash
python train_mimic/scripts/train_climb.py \
    --resume logs/rsl_rl/g1_general_climbing/<run>/model_400.pt \
    --max_iterations 800
```

### 日志

```bash
tensorboard --logdir logs/rsl_rl/g1_general_climbing
```

## 回放

```bash
python train_mimic/scripts/play_climb.py \
    --checkpoint logs/rsl_rl/g1_general_climbing/<run>/model_800.pt \
    --easy --device cuda:0

MUJOCO_GL=egl python train_mimic/scripts/play_climb.py \
    --checkpoint logs/rsl_rl/g1_general_climbing/<run>/model_800.pt \
    --easy --video --device cuda:0
```

## 自动化测试

```bash
python scripts/dev/run_climbing_regression.py --quick
pytest tests/test_climbing_stage*.py -v
```

## 已知限制

- 尚无生产级深度相机；v1 使用 privileged 相对横档几何。
- 仅仿真验证，非 sim2real 部署承诺。
- 梯子拓扑编译一次固定；复位时只改位姿与 mask。
- Easy 预设关闭课程随机化。

## 明确非目标

- 生产 RGB-D / 深度感知
- Sim2real 爬梯部署
- 脚部附着约束
- 每 episode 重编译 MJCF
- 保留 167D 追踪 ONNX 观测合约
