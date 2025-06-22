# cli.py — Entry point to start and manage Temporal workflows

import asyncio
from datetime import timedelta
from temporalio.client import Client
from dataclasses import asdict
from workflows.training_loop_workflow import TrainingWorkflowLoop, TrainingConfig
from workflows.scheduler_workflow import SchedulerWorkflow
from workflows.test_workflow import TrainingWorkflow

# Default training configuration
default_config = TrainingConfig(
    iterations=10,
    checkpoint_dir="ppo_training_scheduler_checkpoint",
    env_name="env.training_scheduler_env.TrainingJobSchedulingEnv",
    num_gpus=0,
    train_batch_size=4000
)

async def run():
    # Connect to Temporal
    client = await Client.connect("localhost:7233")  # or your Temporal Cloud endpoint

    while True:
        print("\nSelect an option:")
        print("1. Start Training Loop Workflow")
        print("2. Stop Training Loop Workflow")
        print("3. Simulate Inference (SchedulerWorkflow)")
        print("4. Test All Activities (SchedulerWorkflow)")
        print("5. Exit")

        choice = input("> ")

        if choice == "1":
            handle = await client.start_workflow(
                TrainingWorkflowLoop.run,
                default_config,
                id="rl-training-loop",
                task_queue="ml-scheduler-task-queue",
            )
            print(f"✅ Training loop started with workflow ID: {handle.id}")

        elif choice == "2":
            try:
                handle = client.get_workflow_handle("rl-training-loop")
                await handle.signal("stop_training")
                print("🛑 Sent stop signal to training loop.")
            except Exception as e:
                print(f"❌ Failed to signal training workflow: {e}")

        elif choice == "3":
            handle = await client.start_workflow(
                SchedulerWorkflow.run,
                id="scheduler-inference-run",
                task_queue="ml-scheduler-task-queue",
                execution_timeout=timedelta(minutes=10),
            )
            print(f"🚀 Inference workflow started with ID: {handle.id}")
            result = await handle.result()
            print(f"🎯 Workflow result: {result}")

        elif choice == "4":
            handle = await client.start_workflow(
                TrainingWorkflow.run,
                default_config,
                id="scheduler-test-run",
                task_queue="ml-scheduler-task-queue",
                execution_timeout=timedelta(minutes=10),
            )
            print(f"🔬 Test workflow started with ID: {handle.id}")
            result = await handle.result()
            print(f"✅ Workflow result: {result}")

        elif choice == "5":
            print("👋 Exiting.")
            break

        else:
            print("❓ Invalid choice. Try again.")

if __name__ == "__main__":
    asyncio.run(run())
