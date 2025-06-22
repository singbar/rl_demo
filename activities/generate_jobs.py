from temporalio import activity
from utils.job_generator import generate_pending_jobs

@activity.defn
async def generate_jobs_activity():
    return generate_pending_jobs()
