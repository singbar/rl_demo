# activities/generate_cluster.py
from temporalio import activity
from utils.job_generator import generate_cluster_state
from utils.to_builtin import to_builtin

@activity.defn
async def generate_cluster_activity(debug: bool = False):
    return to_builtin(generate_cluster_state(debug=debug))