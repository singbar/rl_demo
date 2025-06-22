from temporalio import activity
from utils.job_generator import generate_cluster_state

@activity.defn
async def generate_cluster_activity(_: object=None):
    return generate_cluster_state()