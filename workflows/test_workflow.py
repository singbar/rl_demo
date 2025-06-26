# --- Temporal + RL Training Workflow ---
# This workflow performs a single reinforcement learning training iteration.
# It trains a PPO policy using simulated job and cluster state, then runs inference to validate.

from temporalio import workflow
from dataclasses import dataclass, asdict
from utils.state_encoder import encode_state            # Converts job/cluster state into neural-net inputs
from utils.numpy_converter import convert_numpy         # Converts NumPy arrays to JSON-serializable types
from datetime import timedelta

# Import registered activity functions
from activities.generate_jobs import generate_jobs_activity
from activities.generate_clusuter import generate_cluster_activity
from activities.train import train_policy_activity
from activities.run_policy import run_policy_activity
from activities.apply_schedule_activity import apply_schedule_activity


# --- Config Class Passed into Workflow ---
@dataclass
class TrainingConfig:
    iterations: int = 10                    # Number of PPO training iterations
    checkpoint_dir: str = "ppo_training_scheduler_checkpoint"  # Where to save model checkpoints
    env_name: str = "env.training_scheduler_env.TrainingJobSchedulingEnv"  # RL environment to train in
    num_gpus: int = 0                       # GPU resources allocated for training (if any)
    train_batch_size: int = 4000            # Batch size of transitions collected per training round


@workflow.defn
class TrainingWorkflow:
    @workflow.run
    async def run(self, config: TrainingConfig) -> dict:
        # --- Step 1: Simulate Cluster State ---
        # Generate synthetic cluster hardware state (CPU/GPU availability, health, etc.)
        cluster = workflow.execute_activity(
            generate_cluster_activity,
            args=[False],  # debug flag (False = no console logs)
            start_to_close_timeout=timedelta(seconds=30),
        )

        # --- Step 2: Generate Synthetic Job Queue ---
        # Simulate a batch of 100–500 incoming jobs with random requirements and deadlines
        jobs = workflow.execute_activity(
            generate_jobs_activity,
            args=[False],  # debug flag
            start_to_close_timeout=timedelta(seconds=30),
        )

        await asyncio.gather(cluster,jobs)

        # --- Step 3: Train PPO Policy on Simulated Data ---
        # Calls into an activity that loads the environment, trains the policy, and saves a checkpoint
        checkpoint_path = await workflow.execute_activity(
            train_policy_activity,
            args=[asdict(config)],  # Convert dataclass to dict for serialization
            start_to_close_timeout=timedelta(minutes=15),  # Training can be slow
        )

        # --- Step 4: Run Inference Using Trained Policy ---
        # Converts state to model input format → runs policy → gets suggested action
        obs = encode_state(jobs, cluster)
        obs = convert_numpy(obs)  # Make sure Temporal can serialize the nested structure

        action = await workflow.execute_activity(
            run_policy_activity,
            args=[obs],
            start_to_close_timeout=timedelta(seconds=30),
        )

        # --- Step 5: Apply the Action to Update State ---
        # Converts jobs, cluster, and action into plain Python for activity input
        safe_jobs = convert_numpy(jobs)
        safe_cluster = convert_numpy(cluster)
        safe_action = convert_numpy(action)

        await workflow.execute_activity(
            apply_schedule_activity,
            args=[safe_jobs, safe_cluster, safe_action],
            schedule_to_close_timeout=timedelta(seconds=15),
        )

        # --- Step 6: Return Outputs for Debugging or Testing ---
