from temporalio import workflow
from dataclasses import dataclass
from typing import Optional

import os

# ---- Config input ----
@dataclass
class TrainingConfig:
    iterations: int = 10
    checkpoint_dir: str = "ppo_training_scheduler_checkpoint"
    env_name: str = "env.training_scheduler_env.TrainingJobSchedulingEnv"  # import path
    num_gpus: int = 0


@workflow.defn
class TrainingWorkflow:
    @workflow.run
    async def run(self, config: TrainingConfig) -> str:
        import ray
        from ray.rllib.algorithms.ppo import PPOConfig
        import importlib

        # Ray init (for single-node)
        ray.init(ignore_reinit_error=True, include_dashboard=False)

        # Dynamically import environment
        module_path, class_name = config.env_name.rsplit(".", 1)
        env_module = importlib.import_module(module_path)
        env_class = getattr(env_module, class_name)

        # Configure PPO
        algo = (
            PPOConfig()
            .environment(env=env_class)
            .framework("torch")
            .resources(num_gpus=config.num_gpus)
            .build()
        )

        # Ensure checkpoint directory exists
        os.makedirs(config.checkpoint_dir, exist_ok=True)

        # Training loop
        for i in range(config.iterations):
            result = algo.train()
            reward = result["episode_reward_mean"]
            workflow.logger.info(f"[Iter {i}] reward: {reward:.2f}")

            # Save checkpoint
            checkpoint_path = algo.save(config.checkpoint_dir)
            workflow.logger.info(f"Saved checkpoint to: {checkpoint_path}")

        return f"Training completed after {config.iterations} iterations."
