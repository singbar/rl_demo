import numpy as np

MAX_JOBS = 500
MAX_NODES = 3
JOB_FEATURES = 8
NODE_FEATURES = 6

def encode_state(pending_jobs, cluster_state):
    """
    Convert job queue + cluster resource state into fixed-size RL observation.
    """
    job_obs = np.zeros((MAX_JOBS, JOB_FEATURES), dtype=np.float32)
    cluster_obs = np.zeros((MAX_NODES, NODE_FEATURES), dtype=np.float32)

    # Encode jobs
    for i, job in enumerate(pending_jobs[:MAX_JOBS]):
        job_obs[i, 0] = job["cpu_required"]
        job_obs[i, 1] = job["gpu_required"]
        job_obs[i, 2] = job["priority"]
        job_obs[i, 3] = job["eta_p5"]
        job_obs[i, 4] = job["eta_p4"] if job["eta_p4"] else 0
        job_obs[i, 5] = 1 if job["preferred_gpu"] == "p5" else 0
        job_obs[i, 6] = 1 if job["preferred_gpu"] == "either" else 0
        job_obs[i, 7] = 1  # mask = 1 (active job)

    # Encode nodes
    for i, node in enumerate(cluster_state[:MAX_NODES]):
        cluster_obs[i, 0] = node["available_cpus"]
        cluster_obs[i, 1] = node["available_p5"]
        cluster_obs[i, 2] = node["available_p4"]
        cluster_obs[i, 3] = 1 if node["status"] == "OK" else 0
        cluster_obs[i, 4] = node["hardware"]["p5_perf"]
        cluster_obs[i, 5] = node["hardware"]["p4_perf"]

    # Optional global state
    global_obs = np.array([
        len(pending_jobs),
        sum(1 for n in cluster_state if n["status"] == "OK")
    ], dtype=np.float32)

    return {
        "job_features": job_obs,
        "cluster_features": cluster_obs,
        "global_features": global_obs
    }
