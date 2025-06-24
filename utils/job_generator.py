# Imports
import random
from datetime import datetime, timedelta

# This function simulates the cluster state (i.e., compute nodes and their resources)
# It returns a list of nodes, each with CPU/GPU capacities, availability, and hardware info.
# If debug=True, it prints the full cluster state for inspection.
def generate_cluster_state(debug=False):
    cluster = []

    ### Create a simulated cluster with 3 compute nodes ###
    for i in range(3):
        # Randomly assign total resources to the node within reasonable bounds
        total_cpus = random.randint(400, 800)       # Total available CPU units
        total_p5 = random.randint(2000, 2500)        # High-end GPU units (p5s)
        total_p4 = random.randint(750, 5000)         # Lower-end GPU units (p4s)

        # Create a node dictionary with all relevant information
        cluster.append({
            "node_id": f"node-{i+1}",                # Unique node identifier (e.g., node-1)
            "total_cpus": total_cpus,                # Static total CPU count
            "total_p5": total_p5,                    # Static total P5 GPU count
            "total_p4": total_p4,                    # Static total P4 GPU count
            "available_cpus": total_cpus,            # Mutable available CPU count (starts full)
            "available_p5": total_p5,                # Mutable available P5 count
            "available_p4": total_p4,                # Mutable available P4 count
            "status": random.choice(["OK", "BUSY", "DOWN"]),  # Node health status
            "hardware": {
                "p5_perf": 1.0,      # Performance multiplier for P5 GPUs (fastest)
                "p4_perf": 0.6       # Performance multiplier for P4 GPUs (slower)
            }
        })

        # Print full cluster contents if debugging is enabled
        if debug:
            print(cluster)

    return cluster

# This function creates a randomized list of pending training jobs,
# each with CPU/GPU requirements, priorities, time constraints, and preferences.
# Returns a list of job dictionaries ready for scheduling simulation.
def generate_pending_jobs(debug=False):
    jobs = []
    current_time = datetime.now()

    # Randomly decide how many jobs to generate (between 100 and 500)
    n = random.randint(100, 500)

    # Create offsets used to generate job timing constraints
    arrival_offset = timedelta(seconds=random.randint(0, 300))       # Job arrival in the recent past
    deadline_offset = timedelta(seconds=random.randint(1500, 4000))  # Job deadline in the near future

    ### Generate synthetic job entries ###
    for i in range(n):
        base_eta = random.randint(600, 1800)                 # Estimated runtime in seconds on P5 (best-case)
        gpu_type = random.choice(["p5", "either"])           # Preferred GPU type (strict or flexible)

        job = {
            "job_id": f"job-{i+1}",                          # Unique job identifier (e.g., job-42)
            "cpu_required": random.randint(1, 150),          # CPU units needed for the job
            "gpu_required": random.randint(1, 1500),         # GPU units needed
            "priority": random.randint(1, 3),                # Priority: 1 (low), 2 (medium), 3 (high)
            "arrival_time": (current_time - arrival_offset).isoformat(timespec='seconds'),
            "deadline": (current_time + deadline_offset).isoformat(timespec='seconds'),
            "preferred_gpu": gpu_type,                       # Which GPU the job wants to run on
            "eta_p5": base_eta,                              # Estimated runtime on P5
            "active": True                                   # Whether this job is still eligible to be scheduled
        }

        jobs.append(job)

    # If debug is enabled, print all generated jobs for inspection
    if debug:
        print("Generated pending jobs:")
        for job in jobs:
            print(job)

    return jobs
