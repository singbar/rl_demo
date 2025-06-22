# Imports
import asyncio
from datetime import timedelta
from workflows.training_loop_workflow import TrainingWorkflowLoop, TrainingConfig
from workflows.scheduler_workflow import SchedulerWorkflow
from workflows.test_workflow import TrainingWorkflow
from temporalio.client import Client

async def run():
    # Connect to the Temporal server
    client = await Client.connect("localhost:7233")

    # Present options
    print("\nSelect option:")
    print("1. Start Training Loop")
    print("2. Stop Training Loop")
    print("3. Simulate Inference")
    print("4. Test all activities")

    choice = input("> ")

    if choice == "1":
        handle = await client.start_workflow(
            TrainingWorkflowLoop.run,
            TrainingConfig(
                iterations=10,
                checkpoint_dir="ppo_training_scheduler_checkpoint",
                env_name="env.training_scheduler_env.TrainingJobSchedulingEnv",
                num_gpus=0,
                train_batch_size=4000,
            ),
            id="rl-training-loop",
            task_queue="ml-scheduler-task-queue",
        )
        print(f"Started training loop with ID: {handle.id}")

    elif choice == "2":
        handle = client.get_workflow_handle("rl-training-loop")
        await handle.signal("stop_training")
        print("Sent stop_training signal.")

    elif choice == "3":
        handle = await client.start_workflow(
            SchedulerWorkflow.run,
            id="scheduler-test-run",
            task_queue="ml-scheduler-task-queue",
            execution_timeout=timedelta(minutes=10),
        )
        print(f"Started SchedulerWorkflow with ID: {handle.id}")
        result = await handle.result()
        print(f"Workflow result: {result}")

    elif choice == "4":
        handle = await client.start_workflow(
            TrainingWorkflow.run,
            id="test-activities-run",
            task_queue="ml-scheduler-task-queue",
            execution_timeout=timedelta(minutes=10),
        )
        print(f"Started SchedulerWorkflow with ID: {handle.id}")
        result = await handle.result()
        print(f"Workflow result: {result}")

if __name__ == "__main__":
    asyncio.run(run())
