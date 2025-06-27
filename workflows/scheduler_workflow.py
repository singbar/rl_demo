from temporalio import workflow
from datetime import timedelta

# ⬇ Import all required activities
from activities.generate_jobs import generate_jobs_activity
from activities.generate_clusuter import generate_cluster_activity
from activities.apply_schedule_activity import apply_schedule_activity
from activities.run_policy import run_policy_activity

# ⬇ Utility modules for preparing data for inference
from utils.state_encoder import encode_state
from utils.numpy_converter import convert_numpy


@workflow.defn
class SchedulerWorkflow:
    """
    This Temporal workflow performs **inference-only job scheduling** using a pre-trained PPO policy.
    It's used to simulate how the model behaves in a production loop and supports eventual integration
    with live job streams and real-time cluster state reporting.
    """

    @workflow.run
    async def run(self):
        # 🚀 Main inference loop: repeat every N seconds
        for _ in range(1):  # 🔁 Use `while True` in production for long-running behavior

            # STEP 1: Simulate or retrieve current jobs
            jobs = await workflow.execute_activity(
                generate_jobs_activity,
                schedule_to_close_timeout=timedelta(seconds=60),
            )

            # STEP 2: Simulate or retrieve current cluster resource state
            cluster = await workflow.execute_activity(
                generate_cluster_activity,
                schedule_to_close_timeout=timedelta(seconds=60),
            )

            # STEP 3: Encode the current state into a model-compatible observation
            # This prepares the data for input into the PPO policy (shapes, types, normalization)
            obs = encode_state(jobs, cluster)
            obs = convert_numpy(obs)  # ✅ Ensures it’s safe to pass as JSON to an activity

            # STEP 4: Invoke the pre-trained RL policy to compute a scheduling action
            action = await workflow.execute_activity(
                run_policy_activity,
                args=[obs],
                schedule_to_close_timeout=timedelta(seconds=60),
            )

            # STEP 5: Re-serialize all return values for the next activity
            # Even if the raw objects came from a Temporal activity, re-serialization avoids type issues
            safe_jobs = convert_numpy(jobs)
            safe_cluster = convert_numpy(cluster)
            safe_action = convert_numpy(action)

            # STEP 6: Simulate applying the model's decision (e.g. job→node assignment)
            # This mimics how a real-world job scheduler might execute the action
            await workflow.execute_activity(
                apply_schedule_activity,
                args=[safe_jobs, safe_cluster, safe_action],
                schedule_to_close_timeout=timedelta(seconds=15),
            )

            # STEP 7: Wait before next scheduling window (can represent polling interval or pacing)
            await workflow.sleep(10)

        # 📦 Workflow ends after 10 iterations (for testing or evaluation purposes)
        # In production, we would use a `while True` loop or an external stop signal
