from temporalio import workflow
from dataclasses import dataclass
from typing import Optional
from datetime import timedelta
from activities.generate_jobs import generate_jobs_activity
from activities.generate_clusuter import generate_cluster_activity
from activities.train import train_policy_activity
from activities.run_policy import run_policy_activity

import os

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
            kwargs={"debug": False},  
            start_to_close_timeout=timedelta(seconds=30),
        )

        jobs = await workflow.execute_activity(
            generate_jobs_activity,
            kwargs={"debug": False},
            start_to_close_timeout=timedelta(seconds=30),
        )

        checkpoint_path = await workflow.execute_activity(
            train_policy_activity,
            args=[asdict(config)],
            start_to_close_timeout=timedelta(minutes=15),
        )

        observation = jobs[0]["features"]
        action = await workflow.execute_activity(
            run_policy_activity,
            args=[observation],
            start_to_close_timeout=timedelta(seconds=30),
        )

        return {
            "checkpoint_path": checkpoint_path,
            "sample_action": action,
        }
