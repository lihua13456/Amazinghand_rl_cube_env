在 RL_demo 目录的两个文件中：

- env_cfg.py:70 (Data/KT_data/codex/AmazingHand-main/Demo/AHSimulation/AHSimulation/AH_Right/RL_demo/env_cfg.py:70) 定义立方体；第 72 行的 sim_utils.CuboidCfg(...) 创建它的几何形状。第 91～93 行以同样方式定义支架。
- env.py:52 (Data/KT_data/codex/AmazingHand-main/Demo/AHSimulation/AHSimulation/AH_Right/RL_demo/env.py:52) 根据上述配置创建 RigidObject；第 60～61 行把立方体和支架注册到仿真场景。

主要改 env_cfg.py (Data/KT_data/codex/AmazingHand-main/Demo/AHSimulation/AHSimulation/AH_Right/RL_demo/env_cfg.py)：

- 立方体位置：第 69 行的 cube_start_pos = (x, y, z)，单位是米。第 87 行用它设置初始位置；每次环境重置时，env.py (Data/KT_data/codex/AmazingHand-main/Demo/AHSimulation/AHSimulation/AH_Right/RL_demo/env.py:211) 也会恢复这个位置。因此要移动立方体，优先改 cube_start_pos。
- 立方体朝向：在第 87 行的 InitialStateCfg 中设置 rot=(w, x, y, z)；这是四元数，例如绕 Z 轴转 90° 为 rot=(0.7071, 0, 0, 0.7071)。
- 机器人初始关节姿态：第 51–52 行的 robot_cfg.init_state.joint_pos，目前所有关节都是 0.0。第 109 行的 neutral_motor_rad 则是运行时 8 个电机的目标角度；只改它，手会从初始关节姿态运动到目标姿态，而不是一开始就处于该姿态。
- 机器人整体位置、朝向：第 52 行 robot_cfg.init_state.pos，也可设置 rot。但这只手通过 fixed_base 固定在世界中；如果要移动或旋转整只手，还必须同步处理 env.py 中的固定关节锚点 (Data/KT_data/codex/AmazingHand-main/Demo/AHSimulation/AHSimulation/AH_Right/RL_demo/env.py:66)，不能只改 pos。


对于你当前这份 USD，我核对了 fixed_base：它的一端没有连接刚体，是世界锚点；锚点原始位置为 (0, 0, 0)，旋转为单位四元数。所以移动整只手时，要让“机器人初始位姿”和“世界锚点”使用相同的平移、旋转。

例如，想把整只手沿 X 移动 2 cm，并绕 Z 轴转 90°：

1. 在 env_cfg.py (Data/KT_data/codex/AmazingHand-main/Demo/AHSimulation/AHSimulation/AH_Right/RL_demo/env_cfg.py:51) 设置：

init_state=ArticulationCfg.InitialStateCfg(
    pos=(0.02, 0.0, 0.0),
    rot=(0.7071, 0.0, 0.0, 0.7071),  # 顺序是 w, x, y, z
    joint_pos={".*": 0.0},
),

2. 在 env.py 的 _place_world_anchors() (Data/KT_data/codex/AmazingHand-main/Demo/AHSimulation/AHSimulation/AH_Right/RL_demo/env.py:64) 中，把机器人位移加到现有锚点计算，并设置锚点旋转：

robot_pos = self.cfg.robot_cfg.init_state.pos
w, x, y, z = self.cfg.robot_cfg.init_state.rot

for index, origin in enumerate(self.scene.env_origins.cpu().tolist()):
    joint = UsdPhysics.FixedJoint.Get(
        stage, f"/World/envs/env_{index}/Robot/joints/fixed_base"
    )
    joint.GetLocalPos0Attr().Set(
        Gf.Vec3f(*(float(source_anchor[i]) + origin[i] + robot_pos[i]
                   for i in range(3)))
    )
    joint.GetLocalRot0Attr().Set(
        Gf.Quatf(w, Gf.Vec3f(x, y, z))
    )

只想平移就保持 rot=(1, 0, 0, 0)；只想旋转就保持 pos=(0, 0, 0)。这里的写法针对你当前 fixed_base 的原始锚点恰好为零；如果以后更换 USD，不能直接照搬旋转部分的锚点位置计算。