# --- Imports ---
from temporalio import activity

# --- Activity Definition ---
# This activity simulates how a job would be scheduled onto a cluster node.
# In production, this would likely interact with Kubernetes, ECS, Slurm, etc.
@activity.defn
async def apply_schedule_activity(jobs, cluster, action):
    """
    Simulated job scheduling activity.
    
    Args:
        jobs (list): List of job dictionaries with metadata like job_id, CPU/GPU needs, etc.
        cluster (list): List of cluster node dictionaries with available resources and metadata.
        action (list or tuple): RL model output, typically a (job_idx, node_idx) pair.

    Returns:
        str: Human-readable message describing the scheduled job.
    """

    # --- Step 1: Decode the action ---
    # The RL policy returns an action as a tuple of indices: (job_index, node_index)
    job_idx, node_idx = action 

    # --- Step 2: Resolve job and node by index ---
    # If the job index is invalid (out of bounds), return "invalid" as fallback.
    job_id = jobs[job_idx]["job_id"] if job_idx < len(jobs) else "invalid"
    node_id = cluster[node_idx]["node_id"] if node_idx < len(cluster) else "invalid"

    # --- Step 3: Log and return result ---
    # This simulates applying the schedule and helps visualize decision making during testing.
    print(f"Scheduling {job_id} on {node_id}")
    return f"Scheduled {job_id} on {node_id}"
