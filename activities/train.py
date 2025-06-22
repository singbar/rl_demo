from temporalio import activity
from env.training_scheduler_env import TrainingJobSchedulingEnv
import os

@activity.defn(name="train_policy_activity")
async def train_policy_activity(config: dict = None) -> str:
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig

    # Initialize Ray if needed
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False)

    # Prepare training configuration
    config = config or {}       
    ppo_config = (
        PPOConfig()
        .environment(env=TrainingJobSchedulingEnv)
        .framework("torch")
        .training(train_batch_size=config.get("train_batch_size", 4000))
        .resources(num_gpus=config.get("num_gpus", 0))
    )
    # Build PPO algorithm
    algo = ppo_config.build()

    # Run training iterations
    num_iters = config.get("num_iters", 5)
    for i in range(num_iters):
        result = algo.train()
        print(f"Iteration {i + 1}/{num_iters}, reward: {result['episode_reward_mean']}")

    # Save checkpoint
    checkpoint_dir = config.get("checkpoint_dir", "./checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    result = algo.save(config_dict["checkpoint_dir"])
    checkpoint_path = result.checkpoint.path

    print(f"Checkpoint saved at: {checkpoint_path}")
    return checkpoint_path
