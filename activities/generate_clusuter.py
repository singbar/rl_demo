from temporalio import activity
from utils.job_generator import generate_cluster_state

@activity.defn
async def generate_cluster_activity():
    return generate_cluster_state()