# --- Imports ---
from temporalio import activity
import numpy as np
from env.training_scheduler_env import TrainingJobSchedulingEnv
from utils.to_builtin import to_builtin

# --- Activity Definition ---
# This activity runs inference using a pre-trained PPO (Proximal Policy Optimization) model from Ray RLlib.
# It takes an RL observation (jobs + cluster state), loads the model, and returns a selected scheduling action.

@activity.defn(name="run_policy_activity")
async def run_policy_activity(observation: dict):
    """
    Perform model inference to determine a scheduling decision.

    Args:
        observation (dict): A serialized observation dictionary containing job, cluster, and global features.
                            This is the output of the encode_state() utility.

    Returns:
        tuple(int, int): A tuple representing (job_index, node_index) — the selected job and target node.
    """

    # --- Why inline imports? ---
    # Temporal Workflows run user code inside a *sandboxed environment*. Certain modules like Ray
    # can cause issues if globally imported. Deferring the import to runtime avoids serialization problems.
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig

    # --- Step 1: Initialize Ray (if needed) ---
    # This activity may run in a new or ephemeral worker process. Ray must be initialized defensively.
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False)

    # --- Step 2: Define the RLlib algorithm configuration ---
    # Set up the environment, framework (PyTorch), and resource usage.
    # We disable multi-GPU for local inference here, and skip the new experimental API stack.
    config = (
        PPOConfig()
        .environment(env=TrainingJobSchedulingEnv)  # Custom Gym-like env
        .framework("torch")                         # Use PyTorch backend
        .experimental(_enable_new_api_stack=False)
        .resources(num_gpus=0)                      # Inference runs on CPU
    )

    # --- Step 3: Build the algorithm and restore from checkpoint ---
    # This instantiates the trained PPO agent and loads the latest weights from disk.
    # In production, this checkpoint path could be dynamic or managed via a registry.
    algo = config.build()
    algo.restore("ppo_training_scheduler_checkpoint")

    # --- Step 4: Run inference to get a scheduling action ---
    # compute_single_action returns a raw action tuple — typically (job_index, node_index)
    action = algo.compute_single_action(observation)

    # --- Step 5: Convert output to built-in Python types ---
    # Temporal cannot serialize NumPy arrays or dtypes by default, so we normalize it.
    return to_builtin(action)
