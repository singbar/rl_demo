# --- Temporal Training Loop Workflow ---
# This workflow continuously trains a reinforcement learning policy using PPO.
# It allows external components (e.g. CLI, admin UI, monitoring job) to signal a graceful stop via Temporal Signals.

from temporalio import workflow
from dataclasses import dataclass, asdict
from datetime import timedelta
from activities.train import train_policy_activity  # Activity that runs PPO training and saves checkpoint


# --- Config Input for Training Session ---
@dataclass
class TrainingConfig:
    iterations: int = 10                             # Number of PPO iterations per activity call
    checkpoint_dir: str = "ppo_training_scheduler_checkpoint"  # Directory to store model checkpoints
    env_name: str = "env.training_scheduler_env.TrainingJobSchedulingEnv"  # Python import path for RL environment
    num_gpus: int = 0                                # Optional GPU usage during training
    train_batch_size: int = 4000                     # Number of transitions collected per iteration


@workflow.defn
class TrainingWorkflowLoop:
    """
    A long-running workflow that repeatedly calls the training activity
    until a 'stop_training' signal is received. Useful for continuous learning setups.
    """

    def __init__(self):
        self.keep_training = True  # Internal flag to control loop execution

    @workflow.run
    async def run(self, config: TrainingConfig) -> dict:
        """
        Starts a loop that repeatedly invokes the training activity.
        Can be stopped gracefully using a Temporal signal.
        """

        i = 0
        last_checkpoint_path = None  # Keeps track of the most recent model checkpoint

        # Infinite training loop until signal is received
        while self.keep_training:
            print(f"Training iteration block {i}")

            # Run training activity with provided configuration
            last_checkpoint_path = await workflow.execute_activity(
                train_policy_activity,
                args=[asdict(config)],                     # Convert config dataclass to dictionary
                start_to_close_timeout=timedelta(minutes=15),  # Long-running training allowed
            )
            i += 1

        # Return final checkpoint and total training blocks completed
        return {
            "stopped_after": i,
            "checkpoint_path": last_checkpoint_path,
        }
    @workflow.signal
    def stop_training(self):
        """
        Signal handler for stopping the training loop externally.
        This sets a flag that causes the `while` loop to exit gracefully.
        """
        self.keep_training = False
