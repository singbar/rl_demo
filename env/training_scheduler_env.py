
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from utils.job_generator import generate_cluster_state, generate_pending_jobs
from utils.state_encoder import encode_state

class TrainingJobSchedulingEnv(gym.Env):
    def __init__(self, config=None):
        super().__init__()

        self.max_jobs = 500
        self.max_nodes = 3
        self.job_feature_size = 8
        self.node_feature_size = 6
        self.global_feature_size = 2

        self.observation_space = spaces.Dict({
            "job_features": spaces.Box(low=0, high=np.inf, shape=(self.max_jobs, self.job_feature_size), dtype=np.float32),
            "cluster_features": spaces.Box(low=0, high=np.inf, shape=(self.max_nodes, self.node_feature_size), dtype=np.float32),
            "global_features": spaces.Box(low=0, high=np.inf, shape=(self.global_feature_size,), dtype=np.float32)
        })

        # Action space: choose job index and node index
        self.action_space = spaces.MultiDiscrete([self.max_jobs, self.max_nodes])

        self.current_jobs = None
        self.cluster_state = None
        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_jobs = generate_pending_jobs()
        self.cluster_state = generate_cluster_state()
        self.state = encode_state(self.current_jobs, self.cluster_state)

        return self.state, {}

    def step(self, action):
        job_idx, node_idx = action

        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        # In a real system: check if job can run on node, update resources
        # For now: random reward based on job priority
        if job_idx < len(self.current_jobs) and node_idx < len(self.cluster_state):
            job = self.current_jobs[job_idx]
            node = self.cluster_state[node_idx]
            reward = float(job["priority"])  # placeholder

        # Refresh state
        self.state = encode_state(self.current_jobs, self.cluster_state)

        return self.state, reward, terminated, truncated, info
