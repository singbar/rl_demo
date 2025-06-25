# --- Imports ---
from temporalio import activity
from env.training_scheduler_env import TrainingJobSchedulingEnv
import os

# --- Activity Definition ---
# This activity performs model training using RLlib's PPO algorithm.
# It supports checkpoint loading/saving, configurable hyperparameters, and prints out reward trends.
@activity.defn(name="train_policy_activity")
async def train_policy_activity(config: dict = None) -> str:
    """
    Train a PPO policy on the training scheduler environment.

    Args:
        config (dict): Training configuration parameters (iterations, batch size, GPU count, etc.)

    Returns:
        str: Filesystem path to the final saved checkpoint.
    """

    # --- Inline Import to Bypass Temporal Sandbox Restrictions ---
    # Ray causes issues if loaded globally in Temporal's workflow sandbox, so we import here.
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig

    # --- Defensive Ray Initialization ---
    # Since activities may run in isolated processes, we ensure Ray is initialized only once.
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False)

    # --- Configuration Guardrail ---
    # In case the activity is called without a config, fall back to defaults.
    config = config if isinstance(config, dict) else {}

    # --- PPO Configuration ---
    # This sets up the RLlib training algorithm with our custom environment.
    ppo_config = (
        PPOConfig()
        .environment(env=TrainingJobSchedulingEnv)  # Our cluster/job Gym-style env
        .framework("torch")                         # Using PyTorch instead of TensorFlow
        .training(train_batch_size=config.get("train_batch_size", 4000))
        .resources(num_gpus=config.get("num_gpus", 0))  # GPU support optional
    )

    # --- Build Algorithm ---
    algo = ppo_config.build()

    # --- Optional: Restore Previous Checkpoint ---
    # This allows incremental training — crucial for long-running systems or curriculum learning.
    try:
        algo.restore("ppo_training_scheduler_checkpoint")
        print("Checkpoint restored successfully.")
    except Exception as e:
        print(f"Could not restore checkpoint: {e}")
        print("Continuing with new policy.")

    # --- Training Loop ---
    # Executes the training algorithm for N iterations. Output includes mean reward per episode.
    num_iters = config.get("iterations", 15)
    for i in range(num_iters):
        result = algo.train()
        # Print reward trend (optional: print full result for debugging)
        print(f"Iteration {i + 1}/{num_iters}, reward: {result['episode_reward_mean']}")

    # --- Save Final Checkpoint ---
    # Ensures model progress is not lost and can be reused by other workflows (e.g., inference).
    checkpoint_dir = config.get("checkpoint_dir", "./checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    result = algo.save(checkpoint_dir)
    checkpoint_path = result.checkpoint.path

    print(f"Checkpoint saved at: {checkpoint_path}")
    return checkpoint_path
