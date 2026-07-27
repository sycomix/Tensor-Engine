#!/usr/bin/env python3
"""Small, framework-independent evaluation task harness."""

from __future__ import annotations

import argparse
import os
import random
import sys
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.utils.logging import ExperimentLogger


class EvalTask(ABC):
    """Interface implemented by evaluation tasks."""

    def __init__(self, name: str, config: dict[str, Any]):
        self.name = name
        self.config = config

    @abstractmethod
    def setup(self) -> None:
        """Prepare data, model, and other task resources."""

    @abstractmethod
    def run(self, logger: ExperimentLogger) -> dict[str, Any]:
        """Execute the evaluation and return serializable results."""


class EvalHarness:
    """Register and run reproducible evaluation tasks."""

    def __init__(self, output_dir: str = "logs"):
        self.output_dir = output_dir
        self.tasks: dict[str, EvalTask] = {}

    def register_task(self, task: EvalTask) -> None:
        if task.name in self.tasks:
            raise ValueError(f"Task {task.name} already registered.")
        self.tasks[task.name] = task

    def run_task(self, task_name: str, seed: int = 42) -> dict[str, Any]:
        if task_name not in self.tasks:
            raise ValueError(f"Task {task_name} not found.")

        random.seed(seed)
        np.random.seed(seed)

        task = self.tasks[task_name]
        logger = ExperimentLogger(self.output_dir, task_name)
        logger.info(f"Starting task: {task_name}")
        logger.info(f"Seed: {seed}")
        logger.info(f"Config: {task.config}")

        task.setup()
        results = task.run(logger)
        results["seed"] = seed
        logger.log_result(results)
        logger.info(f"Task {task_name} completed successfully.")
        return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluation Harness")
    parser.add_argument("--task", required=True, choices=("gsm8k_mini",))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="logs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    harness = EvalHarness(args.output_dir)

    if args.task == "gsm8k_mini":
        from scripts.tasks.gsm8k_mini import GSM8KMiniTask

        harness.register_task(GSM8KMiniTask({}))

    harness.run_task(args.task, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
