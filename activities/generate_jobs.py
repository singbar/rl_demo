# --- Imports ---
from temporalio import activity
from utils.job_generator import generate_pending_jobs
from utils.to_builtin import to_builtin

# --- Activity Definition ---
# This activity generates a list of synthetic jobs to simulate pending work in a cluster environment.
# Used during both training and inference workflows to populate the job queue.

@activity.defn
async def generate_jobs_activity(debug: bool = False):
    """
    Generate a simulated list of jobs for scheduling.

    Args:
        debug (bool): Enables verbose logging for inspection and testing purposes.

    Returns:
        list[dict]: List of job descriptions with resource requirements and scheduling constraints,
                    converted to JSON-serializable types for Temporal payload transport.
    """
    
    # --- Step 1: Generate synthetic job workload ---
    # Each job includes random CPU/GPU requirements, priority, preferred GPU type,
    # estimated runtime, and deadline. This simulates a real-world scheduling scenario.
    raw_jobs = generate_pending_jobs(debug=debug)

    # --- Step 2: Ensure Temporal-safe serialization ---
    # The Temporal SDK requires activity inputs and outputs to be JSON-serializable.
    # `to_builtin` recursively converts any NumPy data (e.g., float32) to standard Python types.
    return to_builtin(raw_jobs)
