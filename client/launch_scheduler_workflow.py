import asyncio
from datetime import timedelta
from temporalio.client import Client

async def main():
    # Connect to the Temporal server
    client = await Client.connect("localhost:7233")  # or your cloud address

    # Start the workflow
    result = await client.start_workflow(
        workflow="SchedulerWorkflow",  # Matches @workflow.defn class name
        id="scheduler-test-run",
        task_queue="ml-scheduler-task-queue",
        execution_timeout=timedelta(minutes=10),
    )

    print(f"Started workflow with ID: {result.id}")

    # Optionally wait for result
    final_result = await result.result()
    print(f"Workflow result: {final_result}")

if __name__ == "__main__":
    asyncio.run(main())