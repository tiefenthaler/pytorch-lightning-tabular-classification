from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union

import numpy as np
from lightning.pytorch.callbacks import EarlyStopping


class ValPercentageEarlyStopping(EarlyStopping):
    """
    A custom early stopping callback that triggers training stopping based on relative (percentage) improvement
    in the validation metric (e.g., loss or accuracy). This class supports both minimizing (for loss metrics)
    and maximizing (for accuracy metrics) the monitored metric.

    Args:
        patience (int): The number of epochs with no improvement after which training will be stopped.
            Defaults to 5.
        min_delta_percentage (float): The minimum percentage improvement required in the validation metric
            to reset the early stopping counter. For example, if set to 0.01, at least a 1% improvement is required.
            Defaults to 0.01 (1%).
        monitor (str): The metric to monitor for early stopping (e.g., 'val_loss', 'val_accuracy').
            Defaults to 'val_loss'.
        mode (str): Whether to minimize ('min') or maximize ('max') the monitored metric. Defaults to 'min'.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta_percentage: float = 0.01,
        monitor: str = "val_loss",
        mode: Literal["min", "max"] = "min",
    ) -> None:
        super().__init__(monitor=monitor, patience=patience, min_delta=0.0, mode=mode)
        self.min_delta_percentage = min_delta_percentage
        self.best_metric = np.inf if mode == "min" else -np.inf
        self.epochs_without_improvement = 0
        self.mode = mode

    def on_validation_end(self, trainer, pl_module) -> None:
        """
        Called after each validation epoch. This method checks if the validation metric has improved by the
        required percentage compared to the best observed value. If no improvement is seen over a given number
        of epochs (`patience`), training will stop.

        Args:
            trainer (pl.Trainer): The trainer instance.
            pl_module (pl.LightningModule): The Lightning model being trained.
        """
        # Get the current validation metric (loss or accuracy)
        val_metric = trainer.callback_metrics.get(self.monitor)

        if val_metric is None:
            return

        # Calculate the percentage improvement from the previous best metric
        if self.mode == "min":  # For loss, we are minimizing
            improvement = (
                (self.best_metric - val_metric) / self.best_metric
                if self.best_metric != np.inf
                else 0
            )
        elif self.mode == "max":  # For accuracy, we are maximizing
            improvement = (
                (val_metric - self.best_metric) / self.best_metric
                if self.best_metric != -np.inf
                else 0
            )
        else:
            raise ValueError("Mode must be either 'min' or 'max'.")

        # If the improvement is greater than the specified min_delta_percentage, reset the counter
        if improvement > self.min_delta_percentage:
            self.best_metric = val_metric
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        # If no improvement for the patience number of epochs, stop training
        if self.epochs_without_improvement >= self.patience:
            self.stopped_epoch = trainer.current_epoch
            self._stop_training = True

        # Call the parent class to apply the usual early stopping mechanism
        super().on_validation_end(trainer, pl_module)
