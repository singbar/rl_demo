# --- Cluster & Job Generator Debug Harness ---
# This script directly tests and displays the output of cluster and job generators,
# which simulate the environment for reinforcement learning agents.

import utils.job_generator as generator  # Contains logic to generate synthetic job queues and compute node states

if __name__ == "__main__":
    # --- Generate Cluster State ---
    # Simulates a small compute cluster (3 nodes) with randomized CPU/GPU resources,
    # hardware capabilities, and operational status ("OK", "BUSY", "DOWN").
    # If debug=True, the function will print the generated state for inspection.
    print("Generating Cluster:")
    generator.generate_cluster_state(debug=True)

    # --- Generate Job Queue ---
    # Creates a randomized batch of pending training jobs (between 100 and 500),
    # each with randomized resource demands, priority, deadlines, and GPU preferences.
    # With debug=True, it prints the full list of jobs.
    print("Generating Jobs:")
    generator.generate_pending_jobs(debug=True)
