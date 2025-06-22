from temporalio import activity
import numpy as np


from env.training_scheduler_env import TrainingJobSchedulingEnv
import numpy as np

def to_builtin(x):
    if isinstance(x, np.ndarray):
        return x.tolist()  # handles both 1D and multi-D arrays
    elif isinstance(x, np.generic):
        return x.item()
    else:
        return x

@activity.defn
async def run_policy_activity(observation):
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    
    # Dummy reinit for single-node
    ray.init(ignore_reinit_error=True, include_dashboard=False)

    config = (
        PPOConfig()
        .environment(env=TrainingJobSchedulingEnv)
        .framework("torch")
        .experimental(_enable_new_api_stack=False)
        .env_runners(num_env_runners=0)
        .resources(num_gpus=0)
    )

    algo = config.build()
    #algo.restore("ppo_training_scheduler_checkpoint")

    action = algo.compute_single_action(observation)
    return to_builtin(action)


