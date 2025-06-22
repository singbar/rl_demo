#This worker handles both inference and training tasks

#Imports
import os
from temporalio.worker import Worker
from temporalio.client import Client
from workflows.scheduler_workflow import SchedulerWorkflow
from  workflows.test_workflow import TrainingWorkflow
from workflows.training_loop_workflow import TrainingWorkflowLoop
from activities.generate_clusuter import generate_cluster_activity
from activities.generate_jobs import generate_jobs_activity 
from activities.apply_schedule_activity import  apply_schedule_activity
from activities.run_policy import run_policy_activity
from activities.train import train_policy_activity

async def main():
    print(">>> Worker main() started")

    address = os.getenv("TEMPORAL_ADDRESS", "localhost:7233") 
    print(f">>> Connecting to Temporal at {address}...")

    client = await Client.connect(address)
    print(">>> Connected to Temporal")

    worker = Worker(
        client,
        task_queue="ml-scheduler-task-queue",
        workflows=[SchedulerWorkflow, TrainingWorkflow, TrainingWorkflowLoop],
        activities=[
            generate_cluster_activity,
            generate_jobs_activity,
            run_policy_activity,
            apply_schedule_activity,
            train_policy_activity
        ],
        max_concurrent_activities=15,
    )

    print(">>> Worker starting polling loop...")
    await worker.run()
