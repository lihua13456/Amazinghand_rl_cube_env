# AmazingHand 立方体旋转 PPO 示例

本目录用 Isaac Lab 搭建机械手旋转立方体的强化学习环境，并用 RSL-RL 的 PPO 训练策略。默认显示 2 个环境，也可以只显示 1 个，适合先在本机检查场景和奖励。

当前任务的成功条件是：立方体保持在机械手附近，绕**世界 Z 轴正向净旋转 360°**。立方体初始放在窄支架上，支架会一直存在。因此，当前版本允许手指推动支架上的立方体旋转；它还不是“完全悬空的手内旋转”任务。

## 1. 启动

使用已有的 `env_isaaclab` 环境。在终端中运行：

```bash
conda activate env_isaaclab
cd /home/wei/Data/KT_data/codex/AmazingHand-main/Demo/AHSimulation/AHSimulation/AH_Right/RL_demo

# 先显示 1 个环境，用零策略动作检查 350 步，不进行训练。
/home/wei/IsaacLab/isaaclab.sh -p train.py --num_envs 1 --smoke_steps 350
```

确认机械手、方块和支架的位置及运动正常后，再进行短程训练：

```bash
/home/wei/IsaacLab/isaaclab.sh -p train.py --num_envs 2 --max_iterations 10
```

使用 W&B 在线查看 PPO 曲线：

```bash
wandb login
/home/wei/IsaacLab/isaaclab.sh -p train.py --num_envs 2 --max_iterations 1000 --wandb
```

`--wandb_project` 可指定项目名，默认是 `amazinghand-cube-rotation`。不加 `--wandb` 时使用本地 TensorBoard 日志。脚本默认使用 CPU PhysX；确认场景稳定后可尝试 `--device cuda:0`。可以加 `--headless` 关闭窗口。`--num_envs` 当前只接受 `1` 或 `2`。

`--smoke_steps` 只检查环境，不启动 PPO，也不上传 W&B。这里的“零动作”指策略输出为零；电机仍会趋向配置中的预设夹持角度，所以手指会运动。

## 2. 场景与数据来源

机器人来自相邻目录的 `usd_collision_grasp/AH_Right_collision.usd`，具有原来的闭环关节和 26 个凸包碰撞体。`asset.py` 每次启动时生成 `assets/AH_Right_rl.usd`：去掉原 USD 内嵌的 PhysicsScene，并把没有角度限制的球关节轴标记改为 Isaac Sim 张量接口支持的 X。**原始碰撞 USD 不会被修改。**

每个环境还包含一个动态立方体、窄支架和地面。立方体边长为 `0.03 m`、质量为 `0.025 kg`，初始中心在 `(0.058, 0.029, 0.120) m`。支架支撑立方体，但与机械手之间的碰撞已过滤。

物理步长为 `0.001 s`，每 20 个物理步更新一次策略，即每个策略步为 `0.02 s`。单回合最长 `12 s`，约 600 个策略步。

## 3. 动作、观测与成功条件

策略输出 8 维动作，每一维对应一个电机关节。动作先限制到 `[-1, 1]`，再映射为以下位置目标（单位：弧度）：

```text
目标角度 = neutral_motor_rad + action_scale_rad × 动作
```

之后代码按关节极限截断目标，并用 `target_smoothing` 平滑。当前 `neutral_motor_rad` 为 `[0.45, -0.45, 0, 0, 0, 0, 0.45, -0.45]`，`action_scale_rad` 为 `0.45`。这些值是**电机目标角度**，实际关节位置仍受接触、闭环约束和力矩上限影响。

策略观测共 38 维：8 个电机位置、8 个电机速度、立方体相对初始位置 3 维、姿态四元数 4 维、线速度 3 维、角速度 3 维、累计旋转进度 1 维，以及当前动作 8 维。

程序逐步计算立方体绕世界 Z 轴的有符号 yaw 变化并累计。反向旋转会抵消进度，因此完整转一圈后即使最终朝向与初始朝向相同，累计进度仍可达到 `2π`。达到 `2π` 且立方体还在任务区域内，判定成功并重置环境。以下情况也会结束回合：

| 条件 | 当前阈值 |
| --- | ---: |
| 立方体中心高度过低 | `z ≤ 0.07 m` |
| 离初始中心的平面距离过大 | `≥ 0.05 m` |
| 立方体倾斜过大 | `≥ 0.60 rad`，约 `34.4°` |
| 回合超时 | `12 s` |

若单步 yaw 变化超过 `0.5 rad`，该步的旋转量不计入进度，以过滤姿态跳变。奖励由旋转进度、接近初始位置、动作变化、倾斜、成功奖励和掉落惩罚组成；具体公式在 `env.py` 的 `_get_rewards()` 中。

## 4. 如何看烟雾测试和训练曲线

烟雾测试每 50 步以及每次重置时打印一次状态：

| 字段 | 含义 |
| --- | --- |
| `turn_deg` | 本回合累计的正向净旋转角度 |
| `z` | 立方体中心高度，单位 m |
| `xy_error` | 立方体距初始中心的平面距离，单位 m |
| `tilt` | 立方体倾斜角，单位度 |
| `motor_error` | 8 个电机实际角度与目标角度的平均误差，单位度 |
| `resets` | 这一策略步内重置的环境数 |

这些数值在自动重置**之前**记录。例如，`step=300` 对应约 6 秒；若同时出现 `reward≈-10` 和 `resets=1`，通常表示立方体离开任务区域而受到掉落惩罚。可用同一行的 `z`、`xy_error`、`tilt` 与上表阈值比较。零动作烟雾测试没有旋转 360° 是正常的；它只用于检查场景。

训练结果保存在 `logs/时间戳/`，包括 `env.yaml`、`ppo.yaml`、PPO 检查点和本地日志。使用 W&B 时，可以查看奖励，以及 `rotation_deg`、`cube_height_m`、`cube_xy_distance_m`、`cube_tilt_deg`、`success_rate_episode` 等指标。`success_rate_episode` 是本次训练至今已结束回合的**累计成功率**，不是最近若干回合的滑动成功率。

## 5. 后续修改位置

| 文件 | 适合修改的内容 |
| --- | --- |
| `env_cfg.py` | 物体尺寸、质量、初始位置、电机目标范围、成功阈值和奖励权重 |
| `env.py` | 观测、奖励公式、终止条件和复位逻辑 |
| `rotation_math.py` | 旋转角度的测量方式；更换旋转轴时也需修改这里 |
| `ppo_cfg.py` | PPO 网络规模、学习率、每次更新的采样步数等 |
| `train.py` | 命令行参数、烟雾测试输出、训练与 W&B 开关 |
| `asset.py` | 从原碰撞 USD 生成训练用机器人副本 |

若要替换成立方体以外的物体，应同时检查新物体的初始姿态、碰撞体、质量、支架位置、旋转轴定义和成功阈值。1～2 个环境适合调试；当前代码与参数**尚未证明 PPO 能学会 360° 旋转**，需要结合画面、重置原因和训练曲线逐步调整。

## 6. 常见提示

- `8 != 56`：Isaac Lab 统计到 56 个关节自由度，而代码只为 8 个电机配置了执行器；其余为被动关节。这条警告符合当前机械手结构。
- `ManagerLiveVisualizer ... Manager does not exist`：这是 Direct 环境与 manager 面板不匹配的提示。当前配置已关闭该面板；不会关闭仿真窗口。
- `Using USD spherical joints with any axis except x ...`：训练用 USD 会将无限位球关节的轴标记设为 X。重新运行脚本会重新生成副本；若仍看到此警告，请确认运行的是本目录最新的 `asset.py`。
- `gpu.foundation.plugin ... IMemoryBudgetManagerFactory`：Isaac Sim 内部性能提示，不表示抓取或奖励判定失败。
- 环境无法启动：先确认已激活 `env_isaaclab`，再检查 `python -c "import numpy, torch, isaaclab, rsl_rl, wandb"`。Isaac Sim 即使使用 CPU PhysX，图形初始化通常仍需要本机 NVIDIA 驱动正常工作。
