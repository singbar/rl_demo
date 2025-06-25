# --- Local PPO Training Script for Job Scheduling Environment ---
# This script trains a PPO agent using Ray RLlib on the custom `TrainingJobSchedulingEnv` environment.
# It is standalone (not dependent on Temporal), and is ideal for:
#   ✅ Debugging the reward function and environment dynamics
#   ✅ Tuning hyperparameters locally
#   ✅ Validating environment logic before deploying inside Temporal activities

# --- Imports ---
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from env.training_scheduler_env import TrainingJobSchedulingEnv  # Custom Gymnasium-compatible environment

# --- Training Entry Point ---
def main():
    # Initialize the Ray runtime. This is needed for RLlib training.
    # `ignore_reinit_error=True` allows reruns in interactive sessions.
    ray.init(ignore_reinit_error=True)

    # Register the custom environment with RLlib's registry.
    # This makes it accessible by name during config setup.
    register_env("training_scheduler", lambda cfg: TrainingJobSchedulingEnv(cfg))

    # Define and configure the PPO algorithm.
    config = (
        PPOConfig()
        .environment(env="training_scheduler", env_config={})  # Can pass custom config dict here
        .framework("torch")  # Choose PyTorch as the DL backend
        .rollouts(num_rollout_workers=0)  # Use a single process (local training, avoids multiprocessing issues)
        .training(train_batch_size=4000)  # How many environment steps per policy update
        .resources(num_gpus=0)  # No GPU use; set to >0 for accelerated training
    )

    # Build the algorithm (this instantiates PPO with the above config)
    algo = config.build()

    # --- Training Loop ---
    # Run training for 10 iterations (episodes per iteration depends on env + batch size)
    for i in range(10):
        result = algo.train()  # Executes a training iteration
        print(f"Iteration {i}: mean reward = {result['episode_reward_mean']}")  # Print summary metric

    # --- Save Checkpoint ---
    # Saves model weights and training state for later restoration (e.g., for inference or continued training)
    checkpoint_dir = algo.save("ppo_training_scheduler_checkpoint")
    print(f"Checkpoint saved at: {checkpoint_dir}")

    # Gracefully shut down Ray runtime
    ray.shutdown()

# --- Entry Point Guard ---
if __name__ == "__main__":
    main()
