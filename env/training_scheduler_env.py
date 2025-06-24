#Imports
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from utils.job_generator import generate_cluster_state, generate_pending_jobs
from utils.state_encoder import encode_state

#This class provides ray with instructions on how to initialize the training/inference environment, 
#what to do at each training step, and what to do when restting context between episodes.

class TrainingJobSchedulingEnv(gym.Env):
    def __init__(self, config=None):
        super().__init__()

        self.max_jobs = 500 #Maximum number of jobs that can be submitted to the observation space
        self.max_nodes = 3 #Maximum number of nodes 
        self.max_episode_steps = 100 #Maxiumum number of steps in an episode
        self.job_feature_size = 8 #Number of properties associated with each job in the queue
        self.node_feature_size = 6 #Number of properties associated with each node in the cluster
        self.global_feature_size = 2 #The number of global features (in this case, the number of jobs that need to be scheduled and number of healthy nodes)
        self.steps_taken = 0 #Counts the number of steps taken during each episode

        #Observation Space
        #In the observation space, we define the shape of the model inputs. 
        #   Job Features is defined as a 500x8 (500 jobs by 8 features) matrix with all numeric data in 32-bit floating point numbers
        #   Cluster features is defined as a 3x6 matrix with all numeric data in 32-bit floating point numbers
        #   Global Features is defined as a 2 property matrix

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
        self.steps_taken = 0
        self.current_jobs = generate_pending_jobs()
        self.cluster_state = generate_cluster_state()
        self.state = encode_state(self.current_jobs, self.cluster_state)
        return self.state, {}

    def step(self, action):
        self.steps_taken += 1

        job_idx, node_idx = action
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        if job_idx < len(self.current_jobs) and node_idx < len(self.cluster_state):
            job = self.current_jobs[job_idx]
            reward = float(job.get("priority", 0.0))  # safer

        if self.steps_taken >= self.max_episode_steps:
            terminated = True

        self.state = encode_state(self.current_jobs, self.cluster_state)
        return self.state, reward, terminated, truncated, info