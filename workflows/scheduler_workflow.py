from temporalio import workflow
from temporalio import activity
from datetime import timedelta
from activities.generate_jobs import generate_jobs_activity
from activities.generate_clusuter import generate_cluster_activity
from activities.apply_schedule_activity import apply_schedule_activity
from activities.run_policy import run_policy_activity
from utils.state_encoder import encode_state
from utils.numpy_converter import convert_numpy

@workflow.defn
class SchedulerWorkflow:
    @workflow.run
    async def run(self):
        for _ in range(5):  # replace with while True in prod
            jobs = await workflow.execute_activity(
                generate_jobs_activity,
                schedule_to_close_timeout=timedelta(seconds=15)
            )
            cluster = await workflow.execute_activity(
                generate_cluster_activity,
                schedule_to_close_timeout=timedelta(seconds=15)
            )

            obs = encode_state(jobs, cluster)
            obs = convert_numpy(obs)  # 🔧 Ensure JSON-serializable

            action = await workflow.execute_activity(
                run_policy_activity,
                args=[obs],
                schedule_to_close_timeout=timedelta(seconds=15)
            )

            # 🔧 Ensure all arguments are JSON-serializable
            safe_jobs = convert_numpy(jobs)
            safe_cluster = convert_numpy(cluster)
            safe_action = convert_numpy(action)

            await workflow.execute_activity(
                apply_schedule_activity,
                args=[safe_jobs, safe_cluster, safe_action],
                schedule_to_close_timeout=timedelta(seconds=15)
            )

            await workflow.sleep(10)
