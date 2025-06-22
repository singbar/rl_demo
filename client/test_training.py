# run_workflow.py
import asyncio
from temporalio.client import Client
from workflows.test_workflow import TrainingWorkflow, TrainingConfig

async def main():
    client = await Client.connect("localhost:7233")  # or cloud endpoint
    result = await client.start_workflow(
        TrainingWorkflow.run,
        TrainingConfig(
            iterations = 10,
            checkpoint_dir = "ppo_training_scheduler_checkpoint",
            env_name ="env.training_scheduler_env.TrainingJobScheduleEnv",
            num_gpus=0,
            train_batch_size = 1000
        ),
        id="rl-training-test-001",
        task_queue="ml-scheduler-task-queue",
    )
    print(f"Started workflow: {result.id}")

asyncio.run(main())