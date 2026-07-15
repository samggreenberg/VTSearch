"""Evaluation framework for VTSearch sorting quality."""

from vtscore.eval.al_strategies import STRATEGIES, ALContext, available_strategies, select_next
from vtscore.eval.config import EVAL_DATASETS, EvalQuery
from vtscore.eval.metrics import compute_metrics
from vtscore.eval.runner import run_eval
from vtscore.eval.visualize import plot_eval_results, plot_voting_iterations
from vtscore.eval.voting_iterations import (
    run_voting_iterations_eval,
    run_voting_iterations_eval_from_pickles,
    simulate_voting_iterations,
)

__all__ = [
    "EVAL_DATASETS",
    "STRATEGIES",
    "ALContext",
    "EvalQuery",
    "available_strategies",
    "compute_metrics",
    "plot_eval_results",
    "plot_voting_iterations",
    "run_eval",
    "run_voting_iterations_eval",
    "run_voting_iterations_eval_from_pickles",
    "select_next",
    "simulate_voting_iterations",
]
