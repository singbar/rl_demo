import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from env.training_scheduler_env import TrainingJobSchedulingEnv

def main():
    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Register your custom environment
    register_env("training_scheduler", lambda cfg: TrainingJobSchedulingEnv(cfg))

    # Create and configure PPO algorithm
    config = (
        PPOConfig()
        .environment(env="training_scheduler", env_config={})  # Add env config if needed
        .framework("torch")
        .rollouts(num_rollout_workers=0)  # Avoid multi-worker issues in local dev
        .training(train_batch_size=4000)
        .resources(num_gpus=0)
    )

    algo = config.build()

    # Train
    for i in range(10):
        result = algo.train()
        print(f"Iteration {i}: mean reward = {result['episode_reward_mean']}")

    # Save checkpoint
    checkpoint_dir = algo.save("ppo_training_scheduler_checkpoint")
    print(f"Checkpoint saved at: {checkpoint_dir}")

    # Shutdown Ray
    ray.shutdown()

if __name__ == "__main__":
    main()
