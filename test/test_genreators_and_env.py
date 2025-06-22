#This script is for testing out the generator functions (both cluster generator and job generator) with the environment policy 
#used by both training and inference activities. This script will resent the environment stae. 


#Imports
import numpy as np
from utils.job_generator import generate_cluster_state, generate_pending_jobs
from utils.state_encoder import encode_state
from env.training_scheduler_env import TrainingJobSchedulingEnv


def test_generators_and_env():

    #Generate Cluster State and Synthetic Job Data
    print("=== Testing Job and Cluster Generators ===")
    jobs = generate_pending_jobs()
    cluster = generate_cluster_state()

    print(f"Generated {len(jobs)} jobs")
    print(f"Generated {len(cluster)} cluster nodes")

    #Encode Cluster and Sythetic Jobs
    print("=== Encoding State ===")
    obs = encode_state(jobs, cluster)
    print("Job feature shape:", obs["job_features"].shape)
    print("Cluster feature shape:", obs["cluster_features"].shape)
    print("Global feature shape:", obs["global_features"].shape)

    #Reset Environment state and print the first action
    print("=== Running Dummy Step in Environment ===")
    env = TrainingJobSchedulingEnv()
    state, _ = env.reset()

    dummy_action = [0, 0]  # Pick first job and first node
    next_state, reward, done, truncated, info = env.step(dummy_action)
    print("Step reward:", reward)
    print("Next state keys:", next_state.keys())

if __name__ == "__main__":
    test_generators_and_env()
