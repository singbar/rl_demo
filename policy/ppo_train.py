import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from env.training_scheduler_env import TrainingJobSchedulingEnv

def main():
    # Register the custom environment
    register_env("training_scheduler", lambda cfg: TrainingJobSchedulingEnv(cfg))

    # Initialize Ray (only if not already initialized)
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # Define PPO configuration (fully compatible with Ray 2.47.1)
    config = PPOConfig()
    config.environment(env="training_scheduler")
    config.framework("torch")
    config.training(train_batch_size=4000)
    config.resources(num_gpus=0)

    # Build the algorithm
    algo = config.build()

    # Run training iterations
    for i in range(10):
        result = algo.train()
        print(f"Iteration {i}: reward = {result['episode_reward_mean']}")

    # Save the trained policy
    algo.save("ppo_training_scheduler_checkpoint")

if __name__ == "__main__":
    main()
