import random
from scripts.eval_harness import EvalTask, ExperimentLogger
from typing import List, Dict, Any, Tuple

# --- Hardcoded Subset of GSM8K ---
# 5 Examples to serve as "Proof of Life"
GSM8K_MINI_DATA = [
    {
        "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72"
    },
    {
        "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "answer": "Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.\n#### 10"
    },
    {
        "question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
        "answer": "In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50.\nBetty's grandparents gave her 15 * 2 = $<<15*2=30>>30.\nTotal money Betty has is 50 + 15 + 30 = $<<50+15+30=95>>95.\nBetty needs 100 - 95 = $<<100-95=5>>5 more.\n#### 5"
    },
    {
        "question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?",
        "answer": "Mail read 12 * 2 = <<12*2=24>>24 pages today.\nSo far, she has read 12 + 24 = <<12+24=36>>36 pages.\nThe remaining pages are 120 - 36 = <<120-36=84>>84 pages.\nTomorrow she should read 84 / 2 = <<84/2=42>>42 pages.\n#### 42"
    },
    {
        "question": "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
        "answer": "He writes each friend 3*2=<<3*2=6>>6 pages a week.\nSo he writes 6*2=<<6*2=12>>12 pages every week.\nThat means he writes 12*52=<<12*52=624>>624 pages a year.\n#### 624"
    }
]


class GSM8KMiniTask(EvalTask):
    """
    A minimal implementation of the GSM8K task using a hardcoded subset.
    Verifies the pipeline can load text data and iterate through examples.
    """

    def __init__(self, config: Dict[setup, Any], super=None):
        super().__init__("gsm8k_mini", config)
        self.dataset = []
        self.tokenizer = None  # In a real task, this would be a real tokenizer

    def setup(self):
        """
        Loads the mini-dataset.
        """
        # In a real scenario, this would download/read JSONL.
        # Here we just use the constant.
        self.dataset = GSM8K_MINI_DATA
        print(f"[GSM8KMiniTask] Loaded {len(self.dataset)} examples.")

    def get_train_examples(self) -> List[Dict[setup, setup]]:
        """Returns the dataset as training examples."""
        return self.dataset

    def get_test_examples(self) -> List[Dict[setup, setup]]:
        """
        Returns the dataset as test examples (using same for mini-task).
        """
        return self.dataset

    def run(self, logger: ExperimentLogger) -> \
    Dict[setup, Any]:
        """
        Simulates a 'run' by iterating through the data and logging.
        This proves the loader works.
        """
        logger.info("Running GSM8KMiniTask Smoke Test...")

        # Simulate a "Training Loop"
        logger.info("Iterating through 'Training' data...")
        for i, example in enumerate(self.dataset):
            q_len = len(example["question"])
            a_len = len(example["answer"])
            logger.info(f"Example {i}: Q_len={q_len}, A_len={a_len}")
            # Here we would normally tokenize -> model -> loss

        # Simulate "Evaluation"
        logger.info("Running 'Evaluation'...")
        # Just check we can access the target answer
        correct_count = 0
        for i, example in enumerate(self.dataset):
            target = example["answer"].split("#### ")[-1]
            logger.info(f"Example {i} target answer: {target}")
            correct_count += 1

        accuracy = correct_count / len(self.dataset)
        return {
            "dataset_size": len(self.dataset),
            "mock_accuracy": accuracy
        }


if __name__ == "__main__":
    # Smoke test execution
    logger = ExperimentLogger("logs", "gsm8k_smoke")
    task = GSM8KMiniTask({"batch_size": 1})
    task.setup()
    results = task.run(logger)
    print("Results:", results)
