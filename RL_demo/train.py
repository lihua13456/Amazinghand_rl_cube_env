"""Run a one/two-env smoke test or PPO training, optionally with W&B."""

import argparse
import importlib.util
import importlib.metadata
from datetime import datetime
from pathlib import Path

if importlib.util.find_spec("numpy") is None:
    raise SystemExit(
        "NumPy is missing from the active environment. Activate env_isaaclab, "
        "then repair its dependencies (Isaac Lab requires numpy<2)."
    )

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="AmazingHand cube rotation with Isaac Lab PPO")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--max_iterations", type=int, default=1000)
parser.add_argument("--smoke_steps", type=int, default=0, help="Step the env with zero actions; do not train")
parser.add_argument("--wandb", action="store_true", help="Send PPO curves to Weights & Biases")
parser.add_argument("--wandb_project", default="amazinghand-cube-rotation")
AppLauncher.add_app_launcher_args(parser)
# parser.set_defaults(device="cuda:0")
parser.set_defaults(device="cpu")
args = parser.parse_args()
# if args.num_envs not in (1, 2):
#     raise ValueError("Local debug version supports --num_envs 1 or 2")
if args.smoke_steps < 0 or args.max_iterations < 1:
    raise ValueError("--smoke_steps must be >= 0 and --max_iterations must be >= 1")
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Isaac Sim and USD modules must be imported after AppLauncher.
import torch

from asset import prepare_robot_usd
from env import CubeRotationEnv
from env_cfg import CubeRotationEnvCfg


def main() -> None:
    robot_usd = prepare_robot_usd()
    env_cfg = CubeRotationEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    # The source grasp demo used CPU PhysX; keep that for the first 1-2 envs.
    # Pass --device cuda:0 only after the scene is stable on the local machine.
    env_cfg.sim.device = args.device
    print(f"Robot asset: {robot_usd}")
    log_dir = None
    if not args.smoke_steps:
        log_dir = Path(__file__).resolve().parent / "logs" / datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir.mkdir(parents=True, exist_ok=True)
        env_cfg.log_dir = str(log_dir)

    env = CubeRotationEnv(cfg=env_cfg)
    try:
        if args.smoke_steps:
            env.reset()
            for step in range(args.smoke_steps):
                _, reward, terminated, truncated, extras = env.step(
                    torch.zeros((args.num_envs, 8), device=env.device)
                )
                resets = terminated | truncated
                if step % 50 == 0 or step == args.smoke_steps - 1 or resets.any():
                    log = extras["log"]  # measured before automatic reset
                    print(
                        f"step={step + 1} reward={reward.mean().item():.3f} "
                        f"turn_deg={log['rotation_deg'].item():.1f} "
                        f"z={log['cube_height_m'].item():.3f}m "
                        f"xy_error={log['cube_xy_distance_m'].item():.3f}m "
                        f"tilt={log['cube_tilt_deg'].item():.1f}deg "
                        f"motor_error={log['motor_error_deg'].item():.1f}deg "
                        f"resets={resets.sum().item()}"
                    )
            return

        from rsl_rl.runners import OnPolicyRunner
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
        from isaaclab.utils.io import dump_yaml
        from ppo_cfg import CubeRotationPPORunnerCfg

        agent_cfg = CubeRotationPPORunnerCfg()
        agent_cfg.max_iterations = args.max_iterations
        agent_cfg.device = env.device
        if args.wandb:
            import wandb  # noqa: F401 - fail early with a clear missing-package error
            agent_cfg.logger = "wandb"
            agent_cfg.wandb_project = args.wandb_project #wandb的project名字
        rsl_version = importlib.metadata.version("rsl-rl-lib")#返回rsl-rl-lib版本
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, rsl_version)#根据当前 RSL-RL 版本，对配置中的旧字段/兼容性问题进行处理。

        dump_yaml(str(log_dir / "env.yaml"), env_cfg)#保存env.yaml文件
        dump_yaml(str(log_dir / "ppo.yaml"), agent_cfg)
        print(f"Training logs: {log_dir}")

        wrapped_env = RslRlVecEnvWrapper(env, clip_actions=1.0)#接口适配器，isaaclab 的env与rl_RL,参数clip_actions=1.0裁剪env.step传入actions范围在正负1
        runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)#创建训练器，触发网络创建（定期保存模型）
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=False)#开始真正训练
        #self.alg.update()梯度更新
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
