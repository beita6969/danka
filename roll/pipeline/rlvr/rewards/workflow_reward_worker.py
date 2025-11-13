"""
Workflow Reward Worker for ROLL

This reward worker evaluates generated workflows by:
1. Parsing the workflow code from LLM output
2. Validating syntactic correctness
3. Executing the workflow on a small validation set
4. Computing reward based on performance + validity - cost
"""

import ast
import json
import os
import re
import sys
import tempfile
import importlib.util
from typing import Dict, List, Tuple, Any
import torch

from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.protocol import DataProto
from roll.utils.logging import get_logger

# Add AFlow to path for evaluation
aflow_path = os.path.join(os.path.dirname(__file__), '../../../../../AFlow')
if aflow_path not in sys.path:
    sys.path.insert(0, aflow_path)


logger = get_logger()


class WorkflowEvaluator:
    """
    Evaluates workflows using AFlow's evaluation framework.

    This class bridges ROLL's reward computation with AFlow's benchmark evaluation.
    """

    def __init__(self, quick_eval_samples: int = 10):
        """
        Args:
            quick_eval_samples: Number of samples for quick evaluation
        """
        self.quick_eval_samples = quick_eval_samples
        self.dataset_evaluators = {}

        # Import AFlow evaluation utilities
        try:
            from scripts.evaluator import get_evaluator, DatasetType
            from scripts.async_llm import create_llm_instance
            self.get_evaluator = get_evaluator
            self.DatasetType = DatasetType
            self.create_llm_instance = create_llm_instance
        except ImportError as e:
            logger.error(f"Failed to import AFlow modules: {e}")
            raise

    def execute_and_evaluate(
        self,
        workflow_code: str,
        prompt_code: str,
        dataset: str,
        llm_config: Dict
    ) -> float:
        """
        Execute workflow on validation set and compute score.

        Args:
            workflow_code: Complete workflow Python code
            prompt_code: Custom prompt definitions
            dataset: Dataset name (GSM8K, MATH, etc.)
            llm_config: LLM configuration for workflow execution

        Returns:
            Score in [0, 1]
        """
        # Create temporary module for workflow
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write workflow and prompt files
            workflow_file = os.path.join(tmpdir, "workflow.py")
            prompt_file = os.path.join(tmpdir, "prompt.py")

            with open(workflow_file, 'w') as f:
                f.write(workflow_code)

            with open(prompt_file, 'w') as f:
                f.write(prompt_code)

            # Load workflow class
            try:
                workflow_class = self._load_workflow_class(workflow_file, dataset)
            except Exception as e:
                logger.warning(f"Failed to load workflow class: {e}")
                return 0.0

            # Create workflow instance
            try:
                workflow_instance = workflow_class(
                    name=f"eval_{dataset}",
                    llm_config=llm_config,
                    dataset=dataset
                )
            except Exception as e:
                logger.warning(f"Failed to create workflow instance: {e}")
                return 0.0

            # Get evaluator for dataset
            evaluator = self._get_dataset_evaluator(dataset)

            # Get validation samples (small subset for quick evaluation)
            eval_data = evaluator.get_validation_samples(n=self.quick_eval_samples)

            # Run evaluation
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                score = loop.run_until_complete(
                    self._evaluate_workflow(workflow_instance, eval_data, evaluator)
                )

                loop.close()
                return score

            except Exception as e:
                logger.warning(f"Evaluation failed: {e}")
                return 0.0

    def _load_workflow_class(self, workflow_file: str, dataset: str):
        """Load workflow class from Python file."""
        spec = importlib.util.spec_from_file_location("workflow_module", workflow_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, 'Workflow'):
            raise AttributeError("Module does not have Workflow class")

        return module.Workflow

    def _get_dataset_evaluator(self, dataset: str):
        """Get or create evaluator for dataset."""
        if dataset not in self.dataset_evaluators:
            evaluator = self.get_evaluator(dataset)
            self.dataset_evaluators[dataset] = evaluator

        return self.dataset_evaluators[dataset]

    async def _evaluate_workflow(self, workflow, eval_data, evaluator):
        """Run workflow on evaluation data."""
        correct = 0
        total = len(eval_data)

        for problem_data in eval_data:
            problem = problem_data.get("problem", "") or problem_data.get("question", "")
            expected_answer = problem_data.get("answer", "") or problem_data.get("ground_truth", "")

            try:
                # Execute workflow
                result, cost = await workflow(problem)

                # Check correctness
                is_correct = evaluator.check_answer(result, expected_answer)

                if is_correct:
                    correct += 1

            except Exception as e:
                logger.debug(f"Workflow execution error: {e}")
                continue

        score = correct / total if total > 0 else 0.0
        return score


class WorkflowRewardWorker(Worker):
    """
    Computes rewards for generated workflows.

    Reward components:
    1. Performance gain: (new_score - parent_score) / parent_score
    2. Validity bonus: 1.0 if code is valid and executable
    3. Cost penalty: Based on workflow complexity

    Final reward = α * performance_gain + β * validity - γ * cost
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Reward weights (from config)
        worker_config = kwargs.get('worker_config', {})
        reward_config = getattr(worker_config, 'reward_config', {})

        self.alpha = reward_config.get('alpha', 10.0)  # Performance weight
        self.beta = reward_config.get('beta', 1.0)     # Validity weight
        self.gamma = reward_config.get('gamma', 0.1)   # Cost weight
        self.quick_eval_samples = reward_config.get('quick_eval_samples', 10)

        # LLM config for workflow execution
        self.exec_llm_config = reward_config.get('exec_llm_config', {
            "model_name": "gpt-4o-mini",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "temperature": 0.0
        })

        # Workflow evaluator
        self.evaluator = WorkflowEvaluator(
            quick_eval_samples=self.quick_eval_samples
        )

        logger.info(f"WorkflowRewardWorker initialized with α={self.alpha}, "
                   f"β={self.beta}, γ={self.gamma}")

    def compute_rewards(self, data: DataProto) -> DataProto:
        """
        Compute rewards for generated workflows.

        Input DataProto:
            - non_tensor_batch["response_text"]: Generated workflow outputs
            - non_tensor_batch["parent_score"]: Parent workflow scores
            - non_tensor_batch["tag"]: Dataset tags

        Output DataProto:
            - response_level_rewards: Computed rewards [batch_size]
            - scores: New workflow scores [batch_size]
        """
        response_texts = data.non_tensor_batch["response_text"]
        parent_scores = data.non_tensor_batch.get("parent_score", [0.0] * len(response_texts))
        datasets = data.non_tensor_batch.get("tag", ["unknown"] * len(response_texts))

        rewards = []
        scores = []

        for i, (response, parent_score, dataset) in enumerate(
            zip(response_texts, parent_scores, datasets)
        ):
            logger.debug(f"Processing response {i+1}/{len(response_texts)} for {dataset}")

            # Parse workflow from response
            parsed = self.parse_workflow_response(response)

            # Validate workflow
            is_valid, validation_details = self.validate_workflow(parsed)

            if not is_valid:
                # Invalid workflow: zero reward
                reward = 0.0
                score = 0.0
                logger.debug(f"Invalid workflow: {validation_details}")
            else:
                # Valid workflow: evaluate performance
                try:
                    score = self.evaluator.execute_and_evaluate(
                        workflow_code=parsed["graph"],
                        prompt_code=parsed["prompt"],
                        dataset=dataset,
                        llm_config=self.exec_llm_config
                    )
                except Exception as e:
                    logger.warning(f"Evaluation error: {e}")
                    score = 0.0

                # Compute reward components
                performance_gain = (score - parent_score) / max(parent_score, 0.01)
                validity_bonus = 1.0
                cost_penalty = self.compute_cost_penalty(parsed)

                # Total reward
                reward = (
                    self.alpha * performance_gain +
                    self.beta * validity_bonus -
                    self.gamma * cost_penalty
                )

                logger.debug(f"Score: {score:.3f} (parent: {parent_score:.3f}), "
                           f"reward: {reward:.3f}")

            rewards.append(reward)
            scores.append(score)

        # Convert to tensors
        response_level_rewards = torch.tensor(rewards, dtype=torch.float32)
        scores_tensor = torch.tensor(scores, dtype=torch.float32)

        logger.info(f"Batch rewards - Mean: {response_level_rewards.mean().item():.3f}, "
                   f"Valid rate: {(scores_tensor > 0).float().mean().item():.2%}")

        return DataProto.from_dict({
            "response_level_rewards": response_level_rewards,
            "scores": scores_tensor,
        })

    def parse_workflow_response(self, response: str) -> Dict[str, str]:
        """
        Parse workflow components from LLM response.

        Expected format:
        <modification>...</modification>
        <graph>...</graph>
        <prompt>...</prompt>

        Returns:
            Dict with keys: modification, graph, prompt
        """
        result = {
            "modification": "",
            "graph": "",
            "prompt": ""
        }

        # Extract each component
        for field in result.keys():
            pattern = rf"<{field}>(.*?)</{field}>"
            match = re.search(pattern, response, re.DOTALL)
            if match:
                result[field] = match.group(1).strip()

        return result

    def validate_workflow(self, parsed: Dict[str, str]) -> Tuple[bool, str]:
        """
        Validate parsed workflow.

        Checks:
        1. All required fields present
        2. Python syntax valid
        3. Contains Workflow class
        4. Contains async __call__ method

        Returns:
            (is_valid, details)
        """
        # Check required fields
        if not parsed["graph"]:
            return False, "Missing graph code"

        graph_code = parsed["graph"]

        # Check Python syntax
        try:
            ast.parse(graph_code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

        # Check for Workflow class
        if "class Workflow" not in graph_code:
            return False, "Missing Workflow class"

        # Check for async __call__ method
        if "async def __call__" not in graph_code:
            return False, "Missing async __call__ method"

        # Check for proper return
        if "return" not in graph_code:
            return False, "Missing return statement"

        return True, "Valid"

    def compute_cost_penalty(self, parsed: Dict[str, str]) -> float:
        """
        Compute cost penalty based on workflow complexity.

        Factors:
        - Number of operators used
        - Code length
        - Nested loops/conditionals

        Returns:
            Penalty score (higher = more complex)
        """
        graph_code = parsed["graph"]

        # Count operators
        operator_keywords = [
            "await self.custom",
            "await self.programmer",
            "await self.sc_ensemble",
            "await self.test",
            "await self.review",
            "await self.revise",
        ]
        operator_count = sum(graph_code.count(kw) for kw in operator_keywords)

        # Code length (normalized)
        code_length = len(graph_code) / 1000  # Normalize by 1KB

        # Control flow complexity
        control_flow_count = (
            graph_code.count("for ") +
            graph_code.count("while ") +
            graph_code.count("if ")
        )

        # Combined cost
        cost = (
            0.3 * operator_count +
            0.3 * code_length +
            0.4 * (control_flow_count / 10)  # Normalize
        )

        return min(cost, 10.0)  # Cap at 10.0


# Register worker
__all__ = ["WorkflowRewardWorker"]
