import argparse
import contextlib
import os
from collections.abc import Callable
from numbers import Number
import numpy as np
import typing
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import BaseLogger, TensorboardLogger
with contextlib.suppress(ImportError):
    import wandb

VALID_LOG_VALS_TYPE = int | Number | np.number | np.ndarray | float
# It's unfortunate, but we can't use Union type in isinstance, hence we resort to this
VALID_LOG_VALS = typing.get_args(VALID_LOG_VALS_TYPE)

TRestoredData = dict[str, np.ndarray | dict[str, "TRestoredData"]]


class WandbLogger(BaseLogger):
    """Weights and Biases logger that sends data to https://wandb.ai/.

    This logger creates three panels with plots: train, test, and update.
    Make sure to select the correct access for each panel in weights and biases:

    Example of usage:
    ::

        logger = WandbLogger()
        logger.load(SummaryWriter(log_path))
        result = OnpolicyTrainer(policy, train_collector, test_collector,
                                  logger=logger).run()

    :param train_interval: the log interval in log_train_data(). Default to 1000.
    :param test_interval: the log interval in log_test_data(). Default to 1.
    :param update_interval: the log interval in log_update_data().
        Default to 1000.
    :param info_interval: the log interval in log_info_data(). Default to 1.
    :param save_interval: the save interval in save_data(). Default to 1 (save at
        the end of each epoch).
    :param write_flush: whether to flush tensorboard result after each
        add_scalar operation. Default to True.
    :param str project: W&B project name. Default to "tianshou".
    :param str name: W&B run name. Default to None. If None, random name is assigned.
    :param str entity: W&B team/organization name. Default to None.
    :param str run_id: run id of W&B run to be resumed. Default to None.
    :param argparse.Namespace config: experiment configurations. Default to None.
    """

    def __init__(
        self,
        train_interval: int = 1000,
        test_interval: int = 1,
        update_interval: int = 1000,
        info_interval: int = 1,
        save_interval: int = 1000,
        write_flush: bool = True,
        project: str | None = None,
        name: str | None = None,
        entity: str | None = None,
        run_id: str | None = None,
        config: argparse.Namespace | dict | None = None,
        monitor_gym: bool = True,
    ) -> None:

        super().__init__(train_interval=train_interval, test_interval=test_interval,
                    update_interval=update_interval)
        self.last_save_step = -1
        self.save_interval = save_interval
        self.write_flush = write_flush
        self.restored = False
        if project is None:
            project = os.getenv("WANDB_PROJECT", "real-deal")

        self.wandb_run = (
            wandb.init(
                project=project,
                name=name,
                id=run_id,
                resume="allow",
                entity=entity,
                sync_tensorboard=True,
                monitor_gym=monitor_gym,
                config=config,  # type: ignore
            )
            if not wandb.run
            else wandb.run
        )
        # TODO: don't access private attribute!
        self.wandb_run._label(repo="tianshou")  # type: ignore
        self.tensorboard_logger: TensorboardLogger | None = None
        self.writer: SummaryWriter | None = None

    def prepare_dict_for_logging(self, log_data: dict) -> dict[str, VALID_LOG_VALS_TYPE]:
        if self.tensorboard_logger is None:
            raise Exception(
                "`logger` needs to load the Tensorboard Writer before "
                "preparing data for logging. Try `logger.load(SummaryWriter(log_path))`",
            )
        return self.tensorboard_logger.prepare_dict_for_logging(log_data)

    def load(self, writer: SummaryWriter) -> None:
        self.writer = writer
        self.tensorboard_logger = TensorboardLogger(
            writer,
            self.train_interval,
            self.test_interval,
            self.update_interval,
            self.save_interval,
            self.write_flush,
        )

    def write(self, step_type: str, step: int, data: dict[str, VALID_LOG_VALS_TYPE]) -> None:
        if self.tensorboard_logger is None:
            raise RuntimeError(
                "`logger` needs to load the Tensorboard Writer before "
                "writing data. Try `logger.load(SummaryWriter(log_path))`",
            )
        self.tensorboard_logger.write(step_type, step, data)
        #wandb.log(data, step=step)
        wandb.log(data)


    def log_info_data(self, log_data: dict, step: int) -> None:
        """Use writer to log global statistics.

        :param log_data: a dict containing information of data collected at the end of an epoch.
        :param step: stands for the timestep the training info is logged.
        """
        if (
            step - self.last_log_info_step >= self.info_interval
        ):  # TODO: move interval check to calling method
            log_data = self.prepare_dict_for_logging(log_data)
            self.write(f"{DataScope.INFO.value}/epoch", step, log_data)
            self.last_log_info_step = step
            

    def save_data(
        self,
        epoch: int,
        env_step: int,
        gradient_step: int,
        save_checkpoint_fn: Callable[[int, int, int], str] | None = None,
    ) -> None:
        """Use writer to log metadata when calling ``save_checkpoint_fn`` in trainer.

        :param epoch: the epoch in trainer.
        :param env_step: the env_step in trainer.
        :param gradient_step: the gradient_step in trainer.
        :param function save_checkpoint_fn: a hook defined by user, see trainer
            documentation for detail.
        """
        if save_checkpoint_fn and epoch - self.last_save_step >= self.save_interval:
            self.last_save_step = epoch
            checkpoint_path = save_checkpoint_fn(epoch, env_step, gradient_step)

            checkpoint_artifact = wandb.Artifact(
                "run_" + self.wandb_run.id + "_checkpoint",  # type: ignore
                type="model",
                metadata={
                    "save/epoch": epoch,
                    "save/env_step": env_step,
                    "save/gradient_step": gradient_step,
                    "checkpoint_path": str(checkpoint_path),
                },
            )
            checkpoint_artifact.add_file(str(checkpoint_path))
            self.wandb_run.log_artifact(checkpoint_artifact)  # type: ignore

    def restore_data(self) -> tuple[int, int, int]:
        checkpoint_artifact = self.wandb_run.use_artifact(  # type: ignore
            f"run_{self.wandb_run.id}_checkpoint:latest",  # type: ignore
        )
        assert checkpoint_artifact is not None, "W&B dataset artifact doesn't exist"

        checkpoint_artifact.download(
            os.path.dirname(checkpoint_artifact.metadata["checkpoint_path"]),
        )

        try:  # epoch / gradient_step
            epoch = checkpoint_artifact.metadata["save/epoch"]
            self.last_save_step = self.last_log_test_step = epoch
            gradient_step = checkpoint_artifact.metadata["save/gradient_step"]
            self.last_log_update_step = gradient_step
        except KeyError:
            epoch, gradient_step = 0, 0
        try:  # offline trainer doesn't have env_step
            env_step = checkpoint_artifact.metadata["save/env_step"]
            self.last_log_train_step = env_step
        except KeyError:
            env_step = 0
        return epoch, env_step, gradient_step

    def restore_logged_data(self, log_path: str) -> TRestoredData:
        if self.tensorboard_logger is None:
            raise NotImplementedError(
                "Restoring logged data directly from W&B is not yet implemented."
                "Try instantiating the internal TensorboardLogger by calling something"
                "like `logger.load(SummaryWriter(log_path))`",
            )
        return self.tensorboard_logger.restore_logged_data(log_path)
