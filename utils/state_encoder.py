import numpy as np

MAX_JOBS = 500
MAX_NODES = 3
JOB_FEATURES = 8
NODE_FEATURES = 6

def encode_state(pending_jobs, cluster_state):
    """
    Convert job queue + cluster resource state into fixed-size RL observation.
    Pads or truncates job/node data to fixed shape. Ensures no NaNs.
    """
    job_obs = np.zeros((MAX_JOBS, JOB_FEATURES), dtype=np.float32)
    cluster_obs = np.zeros((MAX_NODES, NODE_FEATURES), dtype=np.float32)

    # Encode jobs
    for i, job in enumerate(pending_jobs[:MAX_JOBS]):
        job_obs[i, 0] = float(job.get("cpu_required", 0))
        job_obs[i, 1] = float(job.get("gpu_required", 0))
        job_obs[i, 2] = float(job.get("priority", 0))
        job_obs[i, 3] = float(job.get("eta_p5") or 0)
        
        eta_p4 = job.get("eta_p4")
        job_obs[i, 4] = float(eta_p4) if isinstance(eta_p4, (int, float)) else 0.0
        
        pref = job.get("preferred_gpu", "")
        job_obs[i, 5] = 1.0 if pref == "p5" else 0.0
        job_obs[i, 6] = 1.0 if pref == "either" else 0.0
        job_obs[i, 7] = 1.0  # mask = 1 (active job)

    # Encode nodes
    for i, node in enumerate(cluster_state[:MAX_NODES]):
        cluster_obs[i, 0] = float(node.get("available_cpus", 0))
        cluster_obs[i, 1] = float(node.get("available_p5", 0))
        cluster_obs[i, 2] = float(node.get("available_p4", 0))
        cluster_obs[i, 3] = 1.0 if node.get("status") == "OK" else 0.0

        hw = node.get("hardware", {})
        cluster_obs[i, 4] = float(hw.get("p5_perf", 0))
        cluster_obs[i, 5] = float(hw.get("p4_perf", 0))

    # Encode global state
    global_obs = np.array([
        float(len(pending_jobs)),
        float(sum(1 for n in cluster_state if n.get("status") == "OK"))
    ], dtype=np.float32)

    # Final observation dict
    obs = {
        "job_features": job_obs,
        "cluster_features": cluster_obs,
        "global_features": global_obs
    }

    # Optional: assert no NaNs
    for key, arr in obs.items():
        if np.isnan(arr).any():
            raise ValueError(f"NaNs found in {key}")

    return obs
