from temporalio import activity
from utils.job_generator import generate_pending_jobs
from utils.to_builtin import to_builtin

@activity.defn
async def generate_jobs_activity(_: object=None):
    return to_builtin(generate_pending_jobs())
