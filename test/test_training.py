# run_workflow.py
import asyncio
from temporalio.client import Client
from workflows.training_workflow import TrainingWorkflow

async def main():
    client = await Client.connect("localhost:7233")  # or cloud endpoint
    result = await client.start_workflow(
        TrainingWorkflow.run,
        id="rl-training-test-001",
        task_queue="rl-training-task-queue",
    )
    print(f"Started workflow: {result.id}")

asyncio.run(main())