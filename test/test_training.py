# run_workflow.py
import asyncio
from temporalio.client import Client
from workflows.train_scheduler_workflow import TrainSchedulerWorkflow

async def main():
    client = await Client.connect("localhost:7233")  # or cloud endpoint
    result = await client.start_workflow(
        TrainSchedulerWorkflow.run,
        id="rl-training-test-001",
        task_queue="rl-training-task-queue",
    )
    print(f"Started workflow: {result.id}")

asyncio.run(main())