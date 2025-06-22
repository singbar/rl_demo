import numpy as np
from utils.job_generator import generate_cluster_state, generate_pending_jobs
from utils.state_encoder import encode_state
from rl_scheduler.env.training_scheduler_env import TrainingJobSchedulingEnv

def test_generators_and_env():
    print("=== Testing Job and Cluster Generators ===")
    jobs = generate_pending_jobs()
    cluster = generate_cluster_state()

    print(f"Generated {len(jobs)} jobs")
    print(f"Generated {len(cluster)} cluster nodes")

    print("=== Encoding State ===")
    obs = encode_state(jobs, cluster)
    print("Job feature shape:", obs["job_features"].shape)
    print("Cluster feature shape:", obs["cluster_features"].shape)
    print("Global feature shape:", obs["global_features"].shape)

    print("=== Running Dummy Step in Environment ===")
    env = TrainingJobSchedulingEnv()
    state, _ = env.reset()

    dummy_action = [0, 0]  # Pick first job and first node
    next_state, reward, done, truncated, info = env.step(dummy_action)
    print("Step reward:", reward)
    print("Next state keys:", next_state.keys())

if __name__ == "__main__":
    test_generators_and_env()
