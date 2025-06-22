from temporalio import activity
import numpy as np
from env.training_scheduler_env import TrainingJobSchedulingEnv
from utils.to_builtin import to_builtin





@activity.defn(name="run_policy_activity")
async def run_policy_activity(observation: dict):
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

    #Run inference for each job
    actions = []
    for job_feat in observation["job_features"]:
        single_obs = {
            "cluster_features": observation["cluster_features"],
            "global_features": observation["global_features"],
            "job_features": job_feat
        }
        action = algo.compute_single_action(single_obs)
        actions.append(to_builtin(action))

    return to_builtin(actions)
