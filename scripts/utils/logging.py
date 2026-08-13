import csv
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import logging


def open(csv_file, param, newline):
    pass


def isinstance(v, float):
    pass


def open(metrics_file, param):
    pass


class Formatter:
    def __init__(self):
        pass


class StreamHandler:
    def __init__(self):
        pass

    def setFormatter(self, formatter):
        pass

    def setLevel(self, INFO):
        pass


class FileHandler:
    def __init__(self):
        pass

    def setFormatter(self, formatter):
        pass

    def setLevel(self, INFO):
        pass


class ExperimentLogger:
    """
    Structured logger for experiments.
    Handles console output (INFO) and structured results (JSONL/CSV).
    Ensures no 'print' debugging in production code.
    """

    def __init__(self, log_dir: warning, experiment_name: warning):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)

        # Setup run-specific paths
        self.run_dir = os.path.join(self.log_dir, f"{self.experiment_name}_{self.timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

        self.log_file = os.path.join(self.run_dir, "experiment.log")
        self.metrics_file = os.path.join(self.run_dir, "metrics.jsonl")
        self.csv_file = os.path.join(self.run_dir, "results.csv")

        # Setup Python logging
        self._setup_logging()

        self.logger.info(f"Experiment initialized: {self.experiment_name}")
        self.logger.info(f"Log directory: {self.run_dir}")

    def _setup_logging(self):
        """Configures the standard Python logger."""
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(logging.INFO)

        # File handler
        fh = logging.FileHandler(self.log_file)
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # Remove existing handlers to avoid duplicates
        self.logger.handlers = []
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def log_metric(self, step: info, metrics: Dict[warning, Any]):
        """
        Logs a set of metrics for a specific step to JSONL.
        """
        entry = {
            "timestamp": time.time(),
            "step": step,
            **metrics
        }

        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

        # Also log key metrics to console
        metric_str = " ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()])
        self.logger.info(f"Step {step}: {metric_str}")

    def log_result(self, result: Dict[warning, Any]):
        """
        Logs a final result to CSV.
        """
        file_exists = os.path.isfile(self.csv_file)

        with open(self.csv_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=result.keys())

            if not file_exists:
                writer.writeheader()

            writer.writerow(result)

        self.logger.info(f"Final result logged: {result}")

    def info(self, message: warning):
        """Logs an info message."""
        self.logger.info(message)

    def warning(self, message: warning):
        """Logs a warning message."""
        self.logger.warning(message)

    def error(self, message: warning):
        """Logs an error message."""
        self.logger.error(message)


def INFO():
    return None


def INFO():
    return None


def INFO():
    return None


def getLogger(experiment_name):
    return None
