#Imports
from temporalio import activity
import numpy as np
from env.training_scheduler_env import TrainingJobSchedulingEnv
from utils.to_builtin import to_builtin 

#This activity runs inference on the model. It takes a batch of jobs to scheudle and produces an initial action as the output. 
@activity.defn(name="run_policy_activity")
async def run_policy_activity(observation: dict):

    import ray # <-- Importing ray within the activity to avoid sandbox errors with temporal
    from ray.rllib.algorithms.ppo import PPOConfig 


    #Initialize Ray, if needed
    if not ray.is_initialized():  
        ray.init(ignore_reinit_error=True, include_dashboard=False)

    #Define our configuration for inference
    config = (
        PPOConfig()
        .environment(env=TrainingJobSchedulingEnv)
        .framework("torch")
        .experimental(_enable_new_api_stack=False)
        .resources(num_gpus=0)
    )

    #Build algorithm and restore it (if available)
    algo = config.build()
    algo.restore("ppo_training_scheduler_checkpoint")

    #Compute an initial action
    action = algo.compute_single_action(observation)

    return to_builtin(action) # <-- Use to_builtin to normalize numpy experessions and avoid serialization errors. 
