# --- Manual Test Harness for State Encoding ---
# This script simulates a cluster environment and job queue,
# then encodes them into a fixed-shape observation suitable for reinforcement learning models.

import utils.job_generator as generator       # Contains job and cluster simulation utilities
import utils.state_encoder as encoder         # Contains encoder logic to transform raw state into model input

if __name__ == "__main__":
    # --- Simulate Cluster State ---
    # Generate a list of compute nodes with available CPU/GPU resources, status, and hardware config
    cluster = generator.generate_cluster_state()

    # --- Simulate Pending Jobs ---
    # Create a randomized batch of jobs with varying resource demands, deadlines, and priorities
    jobs = generator.generate_pending_jobs()

    # --- Encode State for RL Agent ---
    # Transform the (jobs, cluster) pair into a structured NumPy dictionary with fixed shapes
    # Output is a dict with keys: "job_features", "cluster_features", and "global_features"
    obs = encoder.encode_state(jobs, cluster)

    # --- Display the Encoded Observation ---
    print(obs)  # You can inspect the shapes and values to verify they match the observation space
