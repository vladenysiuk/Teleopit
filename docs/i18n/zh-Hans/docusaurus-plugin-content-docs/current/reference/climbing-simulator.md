---
sidebar_position: 7
---

# 爬梯仿真器参考

`General-Climbing-G1` **仿真器侧**实现说明：梯子生成、G1 集成、接触、锁扣、观测（几何提供者）、hold pose 与复位顺序。

奖励/终止与 PPO 见 [爬梯 RL 参考](./climbing-rl.md)。使用说明见 [爬梯教程](../tutorials/climbing.md)。

## 架构概览

```text
LadderConfig → build_ladder_spec() → 固定 MJCF（单 mocap 帧 + max_rungs 圆柱）
        ↓
LadderRuntime：复位时批量写 geom_pos / site_pos + 帧位姿
        ├── ClimbContactState（手↔横档 ID）
        ├── ClimbLatchState（connect 等式）
        └── RelativeRungProvider（躯干系横档端点，供观测）

G1 + 手部点球 + PD 关节 @ 200Hz → 策略 @ 50Hz
```

**固定设计：**

1. 横档为**纯圆柱碰撞**，高摩擦；无隐形平面。
2. **编译一次**固定拓扑；复位只采样位姿与 active mask。
3. **单个 mocap 帧**承载所有横档（非每档独立 mocap）。
4. 锁扣目标在**手球外部中心**：`p_target = p_rung_axis − (r_rung + r_hand + δ) e_x`。

## 核心模块

| 路径 | 作用 |
|------|------|
| `config/ladder.py` | 梯子采样参数 |
| `config/robot.py` | 手点几何与碰撞 bit |
| `ladder/generator.py` | 固定 MJCF 梯子 |
| `ladder/state.py` | 复位采样与 runtime |
| `ladder/contacts.py` | 接触 ID 适配器 |
| `ladder/latch.py` | connect 锁扣后端 |
| `ladder/relative_rungs.py` | K=6 相对横档窗口 |
| `ladder/hold_ik.py` | 四接触静态 IK |
| `mdp/actions.py` | 2D latch 动作 + 迟滞 |
| `mdp/observations.py` | 观测项函数 |

## G1 集成

- 资产：`assets/robots/unitree_g1/g1_29dof.xml`
- 手点：`left/right_hand_point`，半径 15mm，腕部偏移 `(0.18, ±0.025, 0)`
- 碰撞分组：手点仅与梯子横档及必要几何碰撞

## 梯子采样

默认 `max_rungs=12`，激活档数 4–10，间距 220–280mm 严格递增。未激活档移至 `inactive_rung_pose`（z=-50m）。

## 接触

每 `(手, rung_id)` 一个接触传感器。`ClimbContactState` 输出 `hand_in_contact`、`hand_rung_id`（无接触为 -1）、`hand_rung_height`。多接触时取**法向力最大**的匹配横档。

## 锁扣

- 动作：29 关节 + 2 latch；`g>0.5` 附着，`g<-0.5` 分离，中间保持。
- 预声明全部 `(手, 横档, site)` 的 `connect` 等式；每手最多激活一条。
- 禁止不 detach 直接换档。
- `LatchConfig.break_force` 默认 **500 N**：每个物理子步读取激活 connect 等式的 `efc.force`，当 `||F||` 超阈值断开该手；`None` 关闭过载断开（仅调试 / hold-pose）。
- 默认 `solref=(0.05, 1.0)` 软化锁扣，限制附着冲量与弹性储能。不可断开的刚性 connect 相当于焊在运动学梯子坐标系上，可能把浮基弹射出去。

## 观测（仿真提供）

**Actor 本体（101D + 10 帧历史）：** 关节 pos/vel、角速度、重力、上一动作、接触/附着 bit、附着档相对高度。

**Actor 梯子：** K=6（骨盆 ladder 相对高度下 2 上 4），每档 7 维（两端点 torso 坐标 + valid）。**非**世界 Z。

**Critic privileged：** 线速度、骨盆高度、梯框位姿、带 mask 的 active 档高度。

`depth` 模式尚未实现；env 构建时 fail-fast。

## Hold pose

`hold_ik.py` 求四接触静态姿态；`--mode hold-pose` 在重力下持握 2.5s 验证。

## 复位顺序

1.  deactivate latch → 2. 清接触 → 3. 重采样梯子 → 4. 清奖励事件记忆 → 5. 课程 physics/pose

`nconmax` 384–768/env，`njmax` 1200–2560/env（easy hold 更高）。

## 扩展点

- 新横档几何：`ladder/geometry.py` 工厂
- 深度相机：实现 `DepthCameraProvider` 后切换 `mode="depth"`
- 不支持每 episode 重编译 MJCF

Stage 门控笔记：`train_mimic/tasks/climbing/STAGE*.md`。
