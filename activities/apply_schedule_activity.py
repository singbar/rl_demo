from temporalio import activity

@activity.defn
async def apply_schedule_activity(jobs, cluster, action):
    job_idx, node_idx = action
    job_id = jobs[job_idx]["job_id"] if job_idx < len(jobs) else "invalid"
    node_id = cluster[node_idx]["node_id"] if node_idx < len(cluster) else "invalid"
    print(f"Scheduling {job_id} on {node_id}")
    return f"Scheduled {job_id} on {node_id}"