# --- Imports ---
from temporalio import activity
from utils.job_generator import generate_cluster_state
from utils.to_builtin import to_builtin

# --- Activity Definition ---
# This activity generates a simulated cluster state for use in an RL-based scheduler.
# It is used during both training and inference workflows to simulate environment dynamics.
@activity.defn
async def generate_cluster_activity(debug: bool = False):
    """
    Generate a simulated cluster state.

    Args:
        debug (bool): Whether to enable verbose output when generating the cluster. Useful for testing.

    Returns:
        list[dict]: Cluster node metadata, converted to plain Python types for Temporal serialization.
    """
    
    # --- Step 1: Generate synthetic cluster nodes ---
    # This simulates compute nodes with CPU/GPU resources, statuses (e.g., OK/BUSY/DOWN), and hardware performance.
    raw_cluster = generate_cluster_state(debug=debug)

    # --- Step 2: Convert to JSON-safe built-in types ---
    # Temporal requires activity return values to be serializable (no NumPy, etc.).
    # The `to_builtin` utility recursively converts NumPy data to plain Python types.
    return to_builtin(raw_cluster)
