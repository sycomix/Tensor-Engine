
import argparse
import random
import numpy as np
import torch
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.utils.logging import ExperimentLogger

class EvalTask(ABC):
    """
    Abstract base class for evaluation tasks.
    """
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config

    @abstractmethod
    def setup(self):
        """
        Prepare dataset, model, and other resources.
        """
        raise RuntimeError("Abstract method must be implemented in subclasses: setup")

    @abstractmethod
    def run(self, logger: ExperimentLogger) -> Dict[str, Any]:
        """
        Execute the evaluation task.
        Returns a dictionary of results.
        """
        raise RuntimeError("Abstract method must be implemented in subclasses: run")

class EvalHarness:
    """
    Main harness for running evaluation tasks.
    """
    def __init__(self, output_dir: str = "logs"):
        self.output_dir = output_dir
        self.tasks = {}

    def register_task(self, task: EvalTask):
        """Registers a task with the harness."""
        if task.name in self.tasks:
            raise ValueError(f"Task {task.name} already registered.")
        self.tasks[task.name] = task

    def run_task(self, task_name: str, seed: int = 42):
        """Runs a specific registered task."""
        if task_name not in self.tasks:
            raise ValueError(f"Task {task_name} not found.")

        # Set global seeds for reproducibility
        self._set_seed(seed)

        task = self.tasks[task_name]
        logger = ExperimentLogger(self.output_dir, task_name)

        logger.info(f"Starting task: {task_name}")
        logger.info(f"Seed: {seed}")
        logger.info(f"Config: {task.config}")

        try:
            task.setup()
            results = task.run(logger)
            results["seed"] = seed
            logger.log_result(results)
            logger.info(f"Task {task_name} completed successfully.")
        except Exception as e:
            logger.error(f"Task {task_name} failed: {str(e)}")
            raise e

    def _set_seed(self, seed: int):
        """Sets random seeds for Python, NumPy, and PyTorch."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation Harness")
    parser.add_argument("--task", type=str, required=True, help="Name of the task to run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="logs", help="Directory for logs")
    # specific configs can be parsed by tasks, or passed as json string
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Example usage (tasks would normally register themselves)
    harness = EvalHarness(args.output_dir)
    
    # Dynamic import based on task name to register it
    # This part would be expanded as we add tasks
    try:
        if args.task == "mano":
            from scripts.tasks.mano import ManoEmulator
            # Mano is currently run via separate script: experiments/run_mano_poc.py
            print("Mano task is currently run via: python experiments/run_mano_poc.py")
             
        elif args.task == "gsm8k_mini":
             from scripts.tasks.gsm8k_mini import GSM8KMiniTask
             task = GSM8KMiniTask({})
             harness.register_task(task)
             harness.run_task("gsm8k_mini", args.seed)
    except ImportError:
        print(f"Could not import module for task: {args.task}")

    # For now, we assume the script is imported/used by specific run scripts
    # or extended to dynamically load tasks.
