# Imports
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from utils.job_generator import generate_cluster_state, generate_pending_jobs
from utils.state_encoder import encode_state
from datetime import datetime, timedelta

# This class provides Ray with instructions on how to initialize the training/inference environment,
# what to do at each training step, and how to reset context between episodes.
class TrainingJobSchedulingEnv(gym.Env):
    def __init__(self, config=None):
        super().__init__()

        self.max_jobs = 500              # Maximum number of jobs in the observation space
        self.max_nodes = 3               # Number of compute nodes
        self.max_episode_steps = 100     # Steps per episode
        self.job_feature_size = 7        # Properties per job
        self.node_feature_size = 6       # Properties per node
        self.global_feature_size = 2     # Global properties: jobs left, healthy nodes
        self.steps_taken = 0             # Step counter

        # Observation Space
        # In the observation space, we define the shape of the model inputs. 
        #   Job Features is defined as a 500x8 (500 jobs by 7 features) matrix with all numeric data in 32-bit floating point numbers
        #   Cluster features is defined as a 3x6 matrix with all numeric data in 32-bit floating point numbers
        #   Global Features is defined as a 2 property matrix
        self.observation_space = spaces.Dict({
            "job_features": spaces.Box(low=0, high=np.inf, shape=(self.max_jobs, self.job_feature_size), dtype=np.float32),
            "cluster_features": spaces.Box(low=0, high=np.inf, shape=(self.max_nodes, self.node_feature_size), dtype=np.float32),
            "global_features": spaces.Box(low=0, high=np.inf, shape=(self.global_feature_size,), dtype=np.float32)
        })

        # Action space
        # In the action space we tell the model what to do in each action. We use the MultiDiscrete action from Gymnasium.
        # This tells the model to select a value within the range provided. However, the model behavior is nondeterministic.
        # It is possible that the model will still choose an invalid value. 
        self.action_space = spaces.MultiDiscrete([self.max_jobs, self.max_nodes])

        # Initialize key attributes. These will be populated at the first reset.
        self.current_jobs = None
        self.cluster_state = None
        self.state = None

    # Reset Behavior 
    # In gymnasium based environments, reset is called at the start of training and the beginning of each episode.
    def reset(self):
        self.steps_taken = 0
        self.current_jobs = generate_pending_jobs()
        self.cluster_state = generate_cluster_state()
        self.state = encode_state(self.current_jobs, self.cluster_state)
        return self.state

    # Step Behavior
    # In gymnasium, step accepts an action from the agent, updates the environment state, and computes a reward
    def step(self, action):
        self.steps_taken += 1
        reward = 0.0
        terminated = False
        info = {}

        job_idx, node_idx = action

        # Validation 1: Is the action within bounds?
        if job_idx >= len(self.current_jobs) or node_idx >= len(self.cluster_state):
            reward = -1.0
        else:
            job = self.current_jobs[job_idx]
            node = self.cluster_state[node_idx]

            # Extract job and node details
            cpu = job.get("cpu_required", 0)
            gpu = job.get("gpu_required", 0)
            preference = job.get("preferred_gpu", "")
            eta = job.get("eta_p5", 0)
            deadline = datetime.fromisoformat(job.get("deadline"))
            priority = job.get("priority", 1)

            free_cpu = node.get("available_cpus", 0)
            free_p5 = node.get("available_p5", 0)
            free_p4 = node.get("available_p4", 0)
            hw = node.get("hardware", {})
            p5_perf = hw.get("p5_perf", 1.0)
            p4_perf = hw.get("p4_perf", 1.0)

            # Validation 2: Is job already scheduled?
            if not job.get("active", True):
                reward = -1.0

            # Validation 3: Is there enough CPU available?
            elif free_cpu < cpu:
                reward = -1.0

            # Validation 4: Is there enough GPU available?
            elif preference == "p5" and free_p5 < gpu:
                reward = -1.0
            elif preference == "either" and (free_p5 + free_p4) < gpu:
                reward = -1.0

            # Validation 5: Can the job meet its deadline?
            else:
                eta_seconds = eta if preference == "p5" else eta * p4_perf
                estimated_end_time = datetime.now() + timedelta(seconds=eta_seconds)
                if estimated_end_time > deadline:
                    reward = -1.0
                else:
                    # Reward is normalized by max priority (3.0)
                    reward = priority / 3.0

                    # Schedule job
                    job["active"] = False

                    # Deduct resource usage from cluster
                    if gpu > free_p5:
                        p5_used = free_p5
                        p4_used = gpu - p5_used
                    else:
                        p5_used = gpu
                        p4_used = 0

                    node["available_cpus"] -= cpu
                    node["available_p5"] -= p5_used
                    node["available_p4"] -= p4_used

        # Termination condition
        if self.steps_taken >= self.max_episode_steps:
            terminated = True

        # Update state after action
        self.state = encode_state(self.current_jobs, self.cluster_state)
        return self.state, reward, terminated, info
