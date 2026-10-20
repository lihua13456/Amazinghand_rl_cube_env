"""All task parameters live here so objects and rewards are easy to replace."""

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from asset import TRAINING_USD


MOTOR_NAMES = (
    "finger1_motor1", "finger1_motor2",
    "finger2_motor1", "finger2_motor2",
    "finger3_motor1", "finger3_motor2",
    "finger4_motor1", "finger4_motor2",
)


@configclass
class CubeRotationEnvCfg(DirectRLEnvCfg):
    # One policy step = 20 physics steps = 0.02 s. Start with 1 or 2 envs.
    seed = 42
    ui_window_class_type = None  # DirectRLEnv has no manager-based UI panels.
    decimation = 20
    episode_length_s = 12.0
    action_space = 8
    observation_space = 38
    state_space = 0
    viewer: ViewerCfg = ViewerCfg(eye=(0.75, 0.65, 0.45), lookat=(0.0, 0.0, 0.10))
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        # Each world-fixed wrist needs its own anchor; two envs do not need PhysX replication.
        num_envs=2, env_spacing=0.6, replicate_physics=False, clone_in_fabric=False
    )
    sim: SimulationCfg = SimulationCfg(
        dt=0.001,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.9),
        physx=PhysxCfg(enable_ccd=True),
    )

    # The collision hand contains the original closed-loop joints and 26 convex hulls.
    #机械手初始状态
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        articulation_root_prim_path="/joints/fixed_base",
        spawn=sim_utils.UsdFileCfg(usd_path=str(TRAINING_USD)),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, -0.01, 0.09-0.01+0.02),
            # Rotate 90 degrees around each environment's X-axis.
            rot=(0.7011, 0.0923, 0.7011, 0.0923),
            joint_pos={".*": 0.0},
        ),
        actuators={
            # The remaining 48 DOFs are passive joints in the closed-loop hand.
            # Isaac Lab will warn that only 8 of 56 DOFs have actuators; this is intentional.
            "motors": ImplicitActuatorCfg(
                joint_names_expr=list(MOTOR_NAMES),
                stiffness=10.0,#电机驱动刚度
                damping=1.0,#阻尼
                effort_limit_sim=0.25,#力矩上限
            )
        },
    )

    # The cube is free and dynamic. Change this asset configuration to use another object.
    #立方体初始值
    cube_size = (0.06, 0.06, 0.06)
    cube_mass_kg = 0.025
    cube_start_pos = (0.058+0.04, -0.1+0.11, 0.031)
    #添加cube随机范围
    # 每次重置时对立方体初始位姿增加扰动
    randomize_cube_on_reset = True
    cube_reset_xy_noise_m = 0.009                 # X、Y各 ±2 mm
    cube_reset_yaw_noise_rad = math.radians(30.0)  # 绕Z轴 ±5°



    #创建正方体------
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.CuboidCfg(
            size=cube_size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                kinematic_enabled=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=8,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=cube_mass_kg),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0005),
            physics_material=RigidBodyMaterialCfg(static_friction=1.2, dynamic_friction=0.9),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.3, 0.08)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=cube_start_pos),
    )

    # A narrow pedestal makes the initial small-env experiment reproducible.
    # The object remains dynamic; it may still fall from the pedestal.
    #创建定义支架-------
    support_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Support",
        spawn=sim_utils.CuboidCfg(
            size=(0.016, 0.016, 0.105),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, disable_gravity=True
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=RigidBodyMaterialCfg(static_friction=0.2, dynamic_friction=0.1),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.45, 0.6)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.058, 0.029, 0.0525)),
    )

    # Actions are position targets in radians: neutral + action_scale * [-1, 1].
    motor_names = MOTOR_NAMES
    # neutral_motor_rad = (math.pi / 2, -math.pi / 2) * 4  # 90 degrees in radians电机角度
    neutral_motor_rad = (0,0) * 4  # 90 degrees in radians电机角度
    # action_scale_rad = 0.45
    action_scale_rad = 1.57
    target_smoothing = 0.2  # new target fraction at each policy step

    # Positive rotation about world Z. 2*pi is a full turn. See rotation_math.py.
    target_rotation_rad = 6.283185307179586
    max_xy_displacement_m = 0.05
    min_object_height_m = 0.01
    max_object_tilt_rad = 0.60
    max_yaw_increment_rad = 0.5  # reject teleports / numerical jumps

    # Reward per policy step. Keep components separate in env.py for editing.
    rotation_progress_weight = 8.0
    proximity_weight = 0.05
    action_rate_penalty_weight = 0.005
    tilt_penalty_weight = 0.1
    success_bonus = 50.0
    drop_penalty = 10.0

