# This worker handles both inference and training tasks for a reinforcement learning (RL) scheduling system
# It connects to Temporal, registers workflows and activities, and runs a polling loop to process tasks

# --- Imports ---
import os
from temporalio.worker import Worker              # Main Temporal worker class
from temporalio.client import Client              # Used to connect to Temporal server

# Import workflows (defined separately)
from workflows.scheduler_workflow import SchedulerWorkflow            # Handles periodic job scheduling with RL policy
from workflows.test_workflow import TestWorkflow                  # Single-run training workflow for PPO
from workflows.training_loop_workflow import TrainingWorkflowLoop     # Batched/looped training workflow

# Import activity functions (stateless units of work)
from activities.generate_clusuter import generate_cluster_activity    # Simulates current cluster state
from activities.generate_jobs import generate_jobs_activity           # Generates a batch of synthetic jobs
from activities.apply_schedule_activity import apply_schedule_activity  # Applies selected action to the environment
from activities.run_policy import run_policy_activity                 # Runs PPO policy inference
from activities.train import train_policy_activity                    # Trains the PPO agent on collected episodes


# --- Main Async Entry Point ---
async def main():
    print(">>> Worker main() started")

    # Fetch Temporal Cloud or local server address from env var, with localhost fallback
    address = os.getenv("TEMPORAL_ADDRESS", "localhost:7233") 
    print(f">>> Connecting to Temporal at {address}...")

    # Establish connection to Temporal server
    client = await Client.connect(address)
    print(">>> Connected to Temporal")

    # Instantiate a Temporal worker that will:
    # - listen on a task queue
    # - process workflow executions
    # - run registered activities
    worker = Worker(
        client,
        task_queue="ml-scheduler-task-queue",  # All workflows/activities are routed here
        workflows=[
            SchedulerWorkflow,         # Inference-only loop: generate state, run policy, apply action
            TestWorkflow,          # Runs a single training iteration (test/debug use)
            TrainingWorkflowLoop       # Long-running training loop (e.g. train PPO across many batches)
        ],
        activities=[
            generate_cluster_activity, # Simulate cluster state
            generate_jobs_activity,    # Generate jobs to schedule
            run_policy_activity,       # Run trained policy to get action
            apply_schedule_activity,   # Apply action to cluster+jobs
            train_policy_activity      # Train the PPO model using RLlib
        ],
        max_concurrent_activities=15,  # Cap parallel activity execution to avoid overload
    )

    print(">>> Worker starting polling loop...")
    
    # Begin polling loop to receive and process tasks from the Temporal server
    await worker.run()
