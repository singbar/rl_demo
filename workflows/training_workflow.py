from temporalio import workflow
from dataclasses import dataclass, asdict
from utils.state_encoder import encode_state
from utils.numpy_converter import convert_numpy
from datetime import timedelta
from activities.generate_jobs import generate_jobs_activity
from activities.generate_clusuter import generate_cluster_activity
from activities.train import train_policy_activity
from activities.run_policy import run_policy_activity
from activities.apply_schedule_activity import apply_schedule_activity

# ---- Config input ----
@dataclass
class TrainingConfig:
    iterations: int = 10
    checkpoint_dir: str = "ppo_training_scheduler_checkpoint"
    env_name: str = "env.training_scheduler_env.TrainingJobSchedulingEnv"  # import path
    num_gpus: int = 0
    train_batch_size: int = 1000

@workflow.defn
class TrainingWorkflow:
    @workflow.run
    async def run(self, config: TrainingConfig) -> dict:
        cluster = await workflow.execute_activity(
            generate_cluster_activity,
            args=[False],  # ✅ Corrected to list
            start_to_close_timeout=timedelta(seconds=30),
        )

        jobs = await workflow.execute_activity(
            generate_jobs_activity,
            args=[False],  # ✅ Corrected to list
            start_to_close_timeout=timedelta(seconds=30),
        )

        checkpoint_path = await workflow.execute_activity(
            train_policy_activity,
            args=[asdict(config)],
            start_to_close_timeout=timedelta(minutes=15),
        )

        # 🔧 Encode and serialize observation for policy inference
        obs = encode_state(jobs, cluster)
        obs = convert_numpy(obs)
        
        actions = await workflow.execute_activity(
            run_policy_activity,
            args=[obs],
            start_to_close_timeout=timedelta(seconds=30),
        )

        for job, action in zip(jobs, actions):
            await workflow.execute_activity(
            apply_schedule_activity,
            args=[
                convert_numpy(job),
                convert_numpy(cluster),
                convert_numpy(action),
                ],
                schedule_to_close_timeout=timedelta(seconds=15),
            )

        return {
            "checkpoint_path": checkpoint_path,
            "sample_action": actions[0] if actions else None,
        }
