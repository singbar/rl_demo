# This function encodes job and cluster state into a fixed-size observation for input into a reinforcement learning model.
# The encoding converts heterogeneous dictionary structures into structured float32 numpy arrays.

# Imports
import numpy as np

# Constants defining fixed sizes for observation encoding
MAX_JOBS = 500             # Maximum number of jobs to represent (padded/truncated as needed)
MAX_NODES = 3              # Fixed number of nodes to track (consistent with simulated cluster)
JOB_FEATURES = 7           # Number of numeric features used to represent each job
NODE_FEATURES = 6          # Number of numeric features used to represent each node

def encode_state(pending_jobs, cluster_state):
    """
    Convert job queue + cluster resource state into a fixed-shape, float32 RL observation.
    Output is a dictionary with 'job_features', 'cluster_features', and 'global_features'.
    All arrays are padded to fixed sizes to support batching and GPU inference.
    """

    # Initialize a (500 x 7) array for job features. All values start as zero.
    job_obs = np.zeros((MAX_JOBS, JOB_FEATURES), dtype=np.float32)

    # Initialize a (3 x 6) array for node features. All values start as zero.
    cluster_obs = np.zeros((MAX_NODES, NODE_FEATURES), dtype=np.float32)

    # --- Encode each job into a feature vector ---
    for i, job in enumerate(pending_jobs[:MAX_JOBS]):
        job_obs[i, 0] = float(job.get("cpu_required", 0))         # CPU units needed
        job_obs[i, 1] = float(job.get("gpu_required", 0))         # GPU units needed
        job_obs[i, 2] = float(job.get("priority", 0))             # Priority: 1 (low) to 3 (high)
        job_obs[i, 3] = float(job.get("eta_p5") or 0)             # Estimated runtime on P5 GPUs
        job_obs[i, 4] = 1.0 if job.get("active") is True else 0.0 # Binary flag for whether job is active
        pref = job.get("preferred_gpu", "")                       # Preferred GPU type
        job_obs[i, 5] = 1.0 if pref == "p5" else 0.0              # One-hot: prefers p5 only
        job_obs[i, 6] = 1.0 if pref == "either" else 0.0          # One-hot: can run on either p5 or p4

    # --- Encode each node into a feature vector ---
    for i, node in enumerate(cluster_state[:MAX_NODES]):
        cluster_obs[i, 0] = float(node.get("available_cpus", 0))           # CPUs currently free
        cluster_obs[i, 1] = float(node.get("available_p5", 0))             # P5 GPUs free
        cluster_obs[i, 2] = float(node.get("available_p4", 0))             # P4 GPUs free
        cluster_obs[i, 3] = 1.0 if node.get("status") == "OK" else 0.0     # Health indicator (1 if OK)
        hw = node.get("hardware", {})                                      # Nested hardware dictionary
        cluster_obs[i, 4] = float(hw.get("p5_perf", 0))                    # Performance multiplier for P5
        cluster_obs[i, 5] = float(hw.get("p4_perf", 0))                    # Performance multiplier for P4

    # --- Encode global features ---
    # Feature 1: number of jobs remaining to schedule
    # Feature 2: number of healthy nodes
    global_obs = np.array([
        float(len(pending_jobs)),                                          # Job backlog count
        float(sum(1 for n in cluster_state if n.get("status") == "OK"))    # Count of healthy nodes
    ], dtype=np.float32)

    # --- Bundle observations into a single dictionary ---
    obs = {
        "job_features": job_obs,               # Shape: (500, 7)
        "cluster_features": cluster_obs,       # Shape: (3, 6)
        "global_features": global_obs          # Shape: (2,)
    }

    # --- Sanity check: ensure no NaNs in output ---
    for key, arr in obs.items():
        if np.isnan(arr).any():
            raise ValueError(f"NaNs found in {key}")

    return obs
