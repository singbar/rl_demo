from temporalio import workflow
from dataclasses import dataclass, asdict
from datetime import timedelta
from activities.train import train_policy_activity


# ---- Config input ----
@dataclass
class TrainingConfig:
    iterations: int = 10
    checkpoint_dir: str = "ppo_training_scheduler_checkpoint"
    env_name: str = "env.training_scheduler_env.TrainingJobSchedulingEnv"  # import path
    num_gpus: int = 0
    train_batch_size: int = 4000

@workflow.defn
class TrainingWorkflowLoop:
    def __init__(self):
        self.keep_training = True

    @workflow.run
    async def run(self, config: TrainingConfig) -> dict:
        # Register stop signal handler
        workflow.set_signal_handler("stop_training", self.stop_training)

        i = 0
        last_checkpoint_path = None
        while self.keep_training:
            print(f"Training iteration block {i}")
            last_checkpoint_path = await workflow.execute_activity(
                train_policy_activity,
                args=[asdict(config)],
                start_to_close_timeout=timedelta(minutes=15),
            )
            i += 1

        return {
            "stopped_after": i,
            "checkpoint_path": last_checkpoint_path,
        }

    def stop_training(self):
        self.keep_training = False