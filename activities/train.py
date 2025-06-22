#Imports
from temporalio import activity
from env.training_scheduler_env import TrainingJobSchedulingEnv
import os

#This activity runs training to improve the performance of the model
@activity.defn(name="train_policy_activity")
async def train_policy_activity(config: dict = None) -> str:

    import ray # <-- Importing ray within the activity to avoid sandbox errors with temporal
    from ray.rllib.algorithms.ppo import PPOConfig

    # Initialize Ray if needed
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False)

    # Prepare training configuration
    config = config if isinstance(config, dict) else {} # <-- Ensures config is a dict to prevent errors      
    ppo_config = (
        PPOConfig()
        .environment(env=TrainingJobSchedulingEnv)
        .framework("torch")
        .training(train_batch_size=config.get("train_batch_size", 4000))
        .resources(num_gpus=config.get("num_gpus", 0))
    )

    # Build PPO algorithm
    algo = ppo_config.build()
    
    #Restore from an existing checkpoint, if possible
    try:
        algo.restore("ppo_training_scheduler_checkpoint")
        print("Checkpoint restored successfully.")
    except Exception as e:
        print(f"Could not restore checkpoint: {e}")
        print("Continuing with new policy.")

    # Run training iterations
    num_iters = config.get("num_iters", 50) 
    for i in range(num_iters):
        result = algo.train()
        print(f"Raw training result: {result}")
        print(f"Iteration {i + 1}/{num_iters}, reward: {result['episode_reward_mean']}")

    # Save checkpoint
    checkpoint_dir = config.get("checkpoint_dir", "./checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    result = algo.save(checkpoint_dir)
    checkpoint_path = result.checkpoint.path

    print(f"Checkpoint saved at: {checkpoint_path}")
    return checkpoint_path
