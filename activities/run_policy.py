from temporalio import activity
import numpy as np
from env.training_scheduler_env import TrainingJobSchedulingEnv


def to_builtin(x):
    """Convert NumPy types to plain Python types for Temporal serialization."""
    if isinstance(x, np.ndarray):
        if x.size == 1:
            return x.item()
        return x.tolist()
    elif isinstance(x, np.generic):
        return x.item()
    else:
        return x


@activity.defn(name="run_policy_activity")
async def run_policy_activity(observation: dict) -> int:
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig

    # Reinit for local Ray (no dashboard)
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False)

    config = (
        PPOConfig()
        .environment(env=TrainingJobSchedulingEnv)
        .framework("torch")
        .experimental(_enable_new_api_stack=False)
        .resources(num_gpus=0)
    )

    algo = config.build()
    algo.restore("ppo_training_scheduler_checkpoint")

    action = algo.compute_single_action(observation)
    return to_builtin(action)
