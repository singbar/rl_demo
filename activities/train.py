from temporalio import activity
import numpy as np

from env.training_scheduler_env import TrainingJobSchedulingEnv

@activity.defn(name="run_inference_activity")
async def run_inference_activity(observation: dict) -> int:
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig

    # Initialize Ray only if needed
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False)

    # Rebuild PPO config to match the one used during training
    config = (
        PPOConfig()
        .environment(env=TrainingJobSchedulingEnv)
        .framework("torch")
        .training(train_batch_size=4000)
        .resources(num_gpus=0)
    )

    # Rebuild algorithm and load checkpoint
    algo = config.build()
    algo.restore("ppo_training_scheduler_checkpoint")

    # Compute the action
    action = algo.compute_single_action(observation)

    # Ensure action is a scalar int
    if isinstance(action, np.ndarray):
        if action.size != 1:
            raise ValueError("Expected a single action, got array with shape {}".format(action.shape))
        action = int(action.item())
    elif isinstance(action, (int, np.integer)):
        action = int(action)
    else:
        raise TypeError(f"Unexpected action type: {type(action)}")

    return action
