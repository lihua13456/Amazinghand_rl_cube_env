"""Direct Isaac Lab environment: rotate a dynamic cube one full turn."""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

from env_cfg import CubeRotationEnvCfg
from rotation_math import wrapped_yaw_delta, yaw_and_tilt


class CubeRotationEnv(DirectRLEnv):
    cfg: CubeRotationEnvCfg

    def __init__(self, cfg: CubeRotationEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Resolve names once; PhysX's internal joint order need not match JSON order.
        motor_ids = []
        for name in cfg.motor_names:
            matches, _ = self.hand.find_joints(name)
            if len(matches) != 1:
                raise RuntimeError(f"Expected one motor named {name}, found {matches}")
            motor_ids.append(matches[0])
        self.motor_ids = motor_ids

        self.neutral_targets = torch.tensor(
            cfg.neutral_motor_rad, device=self.device, dtype=torch.float32
        ).repeat(self.num_envs, 1)
        limits = self.hand.data.soft_joint_pos_limits[:, self.motor_ids]
        self.neutral_targets = torch.maximum(
            torch.minimum(self.neutral_targets, limits[..., 1]), limits[..., 0]
        )
        self.motor_targets = self.neutral_targets.clone()
        self.actions = torch.zeros(self.num_envs, 8, device=self.device)
        self.previous_actions = torch.zeros_like(self.actions)
        self.previous_yaw = torch.zeros(self.num_envs, device=self.device)
        self.turn_rad = torch.zeros(self.num_envs, device=self.device)
        self.turn_delta = torch.zeros(self.num_envs, device=self.device)
        self.cube_tilt = torch.zeros(self.num_envs, device=self.device)
        self.cube_xy_distance = torch.zeros(self.num_envs, device=self.device)
        self.cube_height = torch.zeros(self.num_envs, device=self.device)
        self.in_workspace = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.succeeded = torch.zeros_like(self.in_workspace)
        self.completed_episodes = torch.zeros((), device=self.device)
        self.successful_episodes = torch.zeros((), device=self.device)

    def _setup_scene(self) -> None:
        self.hand = Articulation(self.cfg.robot_cfg)
        self.cube = RigidObject(self.cfg.object_cfg)#创建RigidObject cube
        # self.support = RigidObject(self.cfg.support_cfg)#创建support
        spawn_ground_plane("/World/Ground", GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self._place_world_anchors()
        self.scene.filter_collisions(global_prim_paths=["/World/Ground"])
        self.scene.articulations["hand"] = self.hand
        self.scene.rigid_objects["cube"] = self.cube #注册立方体到仿真环境
        # self.scene.rigid_objects["support"] = self.support#注册支架到仿真环境
        light = sim_utils.DomeLightCfg(intensity=1000.0)
        light.func("/World/Light", light)

    def _place_world_anchors(self) -> None:
        """Place each fixed wrist anchor using the robot pose in its environment."""
        from pxr import Gf, UsdPhysics

        robot_pos = self.cfg.robot_cfg.init_state.pos
        robot_rot = self.cfg.robot_cfg.init_state.rot

        stage = sim_utils.get_current_stage()
        first_joint = UsdPhysics.FixedJoint.Get(
            stage, "/World/envs/env_0/Robot/joints/fixed_base"
        )
        if not first_joint:
            raise RuntimeError("Could not find the hand's fixed wrist joint")
        source_anchor = first_joint.GetLocalPos0Attr().Get()

        for index, origin in enumerate(self.scene.env_origins.cpu().tolist()):
            joint = UsdPhysics.FixedJoint.Get(
                stage, f"/World/envs/env_{index}/Robot/joints/fixed_base"
            )

            joint.GetLocalPos0Attr().Set(
                Gf.Vec3f(
                    float(origin[0] + robot_pos[0] + source_anchor[0]),
                    float(origin[1] + robot_pos[1] + source_anchor[1]),
                    float(origin[2] + robot_pos[2] + source_anchor[2]),
                )
            )

            w, x, y, z = robot_rot
            joint.GetLocalRot0Attr().Set(
                Gf.Quatf(
                    float(w),
                    Gf.Vec3f(float(x), float(y), float(z)),
                )
            )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clamp(-1.0, 1.0)
        desired = self.neutral_targets + self.cfg.action_scale_rad * self.actions
        limits = self.hand.data.soft_joint_pos_limits[:, self.motor_ids]
        desired = torch.maximum(torch.minimum(desired, limits[..., 1]), limits[..., 0])
        self.motor_targets.lerp_(desired, self.cfg.target_smoothing)

    def _apply_action(self) -> None:
        self.hand.set_joint_position_target(self.motor_targets, joint_ids=self.motor_ids)

    def _measure_cube(self) -> tuple[torch.Tensor, torch.Tensor]:
        cube_pos = self.cube.data.root_pos_w
        local_pos = cube_pos - self.scene.env_origins
        yaw, tilt = yaw_and_tilt(self.cube.data.root_quat_w)
        home = torch.tensor(self.cfg.cube_start_pos, device=self.device)
        xy_distance = torch.linalg.vector_norm(local_pos[:, :2] - home[:2], dim=-1)
        self.cube_xy_distance = xy_distance
        self.cube_height = local_pos[:, 2]
        valid = (
            (xy_distance < self.cfg.max_xy_displacement_m)
            & (local_pos[:, 2] > self.cfg.min_object_height_m)
            & (tilt < self.cfg.max_object_tilt_rad)
        )

        self.cube_tilt = tilt
        self.in_workspace = valid
        return yaw, local_pos

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        yaw, _ = self._measure_cube()
        delta = wrapped_yaw_delta(yaw, self.previous_yaw)
        self.previous_yaw = yaw
        # Ignore pose discontinuities and rotations after the cube has left the task area.
        delta = torch.where(
            self.in_workspace & (delta.abs() <= self.cfg.max_yaw_increment_rad),
            delta,
            torch.zeros_like(delta),
        )
        old_turn = self.turn_rad.clone()
        self.turn_rad = (self.turn_rad + delta).clamp_min(0.0)
        self.turn_delta = self.turn_rad - old_turn
        self.succeeded = self.in_workspace & (self.turn_rad >= self.cfg.target_rotation_rad)
        dropped = ~self.in_workspace
        timed_out = self.episode_length_buf >= self.max_episode_length - 1
        return self.succeeded | dropped, timed_out

    def _get_rewards(self) -> torch.Tensor:
        """Edit these terms when changing the task or reward shaping."""
        cube_local_pos = self.cube.data.root_pos_w - self.scene.env_origins
        home = torch.tensor(self.cfg.cube_start_pos, device=self.device)
        distance = torch.linalg.vector_norm(cube_local_pos - home, dim=-1)
        action_change = torch.square(self.actions - self.previous_actions).sum(dim=-1)

        progress_reward = self.cfg.rotation_progress_weight * self.turn_delta
        proximity_reward = self.cfg.proximity_weight * torch.exp(-distance / 0.03)
        action_penalty = self.cfg.action_rate_penalty_weight * action_change
        tilt_penalty = self.cfg.tilt_penalty_weight * self.cube_tilt.square()
        success_reward = self.cfg.success_bonus * self.succeeded.float()
        drop_penalty = self.cfg.drop_penalty * (~self.in_workspace).float()
        reward = (
            progress_reward + proximity_reward + success_reward
            - action_penalty - tilt_penalty - drop_penalty
        )

        completed_now = self.reset_terminated | self.reset_time_outs
        self.completed_episodes += completed_now.float().sum()
        self.successful_episodes += self.succeeded.float().sum()
        #self.extras是对DirectRLEnv里面的属性进行重写
        self.extras["log"] = {
            "rotation_deg": torch.rad2deg(self.turn_rad).mean(),
            "rotation_progress_deg": torch.rad2deg(self.turn_delta).mean(),
            "cube_distance_m": distance.mean(),
            "cube_height_m": self.cube_height.mean(),
            "cube_xy_distance_m": self.cube_xy_distance.mean(),
            "cube_tilt_deg": torch.rad2deg(self.cube_tilt).mean(),
            "motor_error_deg": torch.rad2deg(
                (self.hand.data.joint_pos[:, self.motor_ids] - self.motor_targets).abs()
            ).mean(),
            "success_rate_step": self.succeeded.float().mean(),
            "success_rate_episode": self.successful_episodes / self.completed_episodes.clamp_min(1.0),
            "drop_rate_step": (~self.in_workspace).float().mean(),
            "reward_progress": progress_reward.mean(),
        }
        self.previous_actions.copy_(self.actions)
        return reward

    def _get_observations(self) -> dict[str, torch.Tensor]:
        cube_pos = self.cube.data.root_pos_w - self.scene.env_origins
        home = torch.tensor(self.cfg.cube_start_pos, device=self.device)
        obs = torch.cat(
            (
                self.hand.data.joint_pos[:, self.motor_ids],            # 8
                self.hand.data.joint_vel[:, self.motor_ids],            # 8
                cube_pos - home,                                         # 3
                self.cube.data.root_quat_w,                             # 4
                self.cube.data.root_lin_vel_w,                          # 3
                self.cube.data.root_ang_vel_w,                          # 3
                (self.turn_rad / self.cfg.target_rotation_rad)[:, None],# 1
                self.actions,                                           # 8
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        super()._reset_idx(env_ids)

        # Restore the cube and all robot joints. No reset randomization during
        # initial local debugging; add it here once contact/rotation are verified.
        cube_state = self.cube.data.default_root_state[env_ids].clone()
        num_resets = cube_state.shape[0]
        if self.cfg.randomize_cube_on_reset:
            # 每个待重置环境独立采样X、Y偏移；Z保持不变
            xy_noise = torch.empty(
                (num_resets, 2), device=self.device
            ).uniform_(
                -self.cfg.cube_reset_xy_noise_m,
                self.cfg.cube_reset_xy_noise_m,
            )
            cube_state[:, :2] += xy_noise
            # 只随机绕Z轴的旋转，roll和pitch扰动为0
            yaw_noise = torch.empty(
                num_resets, device=self.device
            ).uniform_(
                -self.cfg.cube_reset_yaw_noise_rad,
                self.cfg.cube_reset_yaw_noise_rad,
            )
            zeros = torch.zeros_like(yaw_noise)
            yaw_quat = quat_from_euler_xyz(zeros, zeros, yaw_noise)

            # 将旋转扰动叠加到默认初始姿态上
            cube_state[:, 3:7] = quat_mul(
            yaw_quat, cube_state[:, 3:7])

        # 记录本次真正的初始朝向，供后面初始化旋转计数使用
        initial_yaw, _ = yaw_and_tilt(cube_state[:, 3:7])
        cube_state[:, :3] += self.scene.env_origins[env_ids]
        # 重置时清零线速度、角速度
        cube_state[:, 7:] = 0.0

        #写入cube环境
        self.cube.write_root_pose_to_sim(cube_state[:, :7], env_ids)#数据代表[x,y,z,qw,qx,qy,qz]
        self.cube.write_root_velocity_to_sim(cube_state[:, 7:], env_ids)#数据代表[vx,vy,vz,wx,wy,wz]

        joint_pos = self.hand.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        self.hand.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.motor_targets[env_ids] = self.neutral_targets[env_ids]
        self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0
        self.previous_yaw[env_ids] = initial_yaw
        self.turn_rad[env_ids] = 0.0
        self.turn_delta[env_ids] = 0.0
        self.cube_tilt[env_ids] = 0.0
        self.cube_xy_distance[env_ids] = 0.0
        self.cube_height[env_ids] = self.cfg.cube_start_pos[2]
        self.in_workspace[env_ids] = True
        self.succeeded[env_ids] = False
