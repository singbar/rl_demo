# --- cli.py — Entry point to start and manage Temporal workflows interactively ---
# This script allows users to:
#   1. Start a long-running PPO training loop
#   2. Stop the training loop via signal
#   3. Simulate inference using the latest policy
#   4. Test all Temporal activities together in a sample workflow
#   5. Exit the program

# --- Imports ---
import asyncio
from datetime import timedelta
from temporalio.client import Client
from dataclasses import asdict

# Import workflow definitions and config dataclass
from workflows.training_loop_workflow import TrainingWorkflowLoop, TrainingConfig
from workflows.scheduler_workflow import SchedulerWorkflow
from workflows.test_workflow import TestWorkflow

# --- Default Config ---
# This is passed to training workflows to configure PPO training.
default_config = TrainingConfig(
    iterations=10,
    checkpoint_dir="ppo_training_scheduler_checkpoint",
    env_name="env.training_scheduler_env.TrainingJobSchedulingEnv",
    num_gpus=0,
    train_batch_size=4000
)

# --- Async Main Menu Loop ---
async def run():
    """
    Connects to Temporal and presents a CLI menu to manage workflows.
    Each menu item triggers one or more Temporal workflow executions or signals.
    """
    client = await Client.connect("localhost:7233")  # Temporal Cloud URL can also go here

    while True:
        # --- Display menu options ---
        print("\nSelect an option:")
        print("1. Start Training Loop Workflow")
        print("2. Stop Training Loop Workflow")
        print("3. Simulate Inference (SchedulerWorkflow)")
        print("4. Test All Activities (TrainingWorkflow)")
        print("5. Exit")

        choice = input("> ")

        # --- Option 1: Start long-running training loop ---
        if choice == "1":
            handle = await client.start_workflow(
                TrainingWorkflowLoop.run,
                default_config,
                id="rl-training-loop",  # Static ID for resuming or signaling later
                task_queue="ml-scheduler-task-queue",
            )
            print(f"✅ Training loop started with workflow ID: {handle.id}")

        # --- Option 2: Stop training via workflow signal ---
        elif choice == "2":
            try:
                handle = client.get_workflow_handle("rl-training-loop")
                await handle.signal("stop_training")
                print("🛑 Sent stop signal to training loop.")
            except Exception as e:
                print(f"❌ Failed to signal training workflow: {e}")

        # --- Option 3: Simulate inference using latest model ---
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

        # --- Option 4: End-to-end test of all activities ---
        elif choice == "4":
            handle = await client.start_workflow(
                TestWorkflow.run,
                default_config,
                id="e2e-test-run",
                task_queue="ml-scheduler-task-queue",
                execution_timeout=timedelta(minutes=10),
            )
            print(f"🔬 Test workflow started with ID: {handle.id}")
            result = await handle.result()
            print(f"✅ Workflow result: {result}")

        # --- Option 5: Exit program ---
        elif choice == "5":
            print("👋 Exiting.")
            break

        # --- Handle invalid input ---
        else:
            print("❓ Invalid choice. Try again.")

# --- Run the main loop if launched directly ---
if __name__ == "__main__":
    asyncio.run(run())  # Starts event loop and CLI interaction
