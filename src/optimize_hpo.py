"""Hyperparameter optimization helpers for CKD model comparison."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
import warnings

import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold


RANDOM_SEED = 42


def make_cv(n_splits: int = 5, random_state: int = RANDOM_SEED) -> StratifiedKFold:
    """Create the default 5-fold stratified CV splitter."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def run_grid_search(
    estimator,
    param_grid: Mapping[str, list[Any]],
    x,
    y,
    scoring: str = "f1",
    n_splits: int = 5,
    n_jobs: int = 1,
) -> GridSearchCV:
    """Run exhaustive Grid Search."""
    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        cv=make_cv(n_splits=n_splits),
        n_jobs=n_jobs,
        refit=True,
    )
    return search.fit(x, y)


def run_random_search(
    estimator,
    param_distributions: Mapping[str, Any],
    x,
    y,
    n_iter: int = 50,
    scoring: str = "f1",
    n_splits: int = 5,
    random_state: int = RANDOM_SEED,
    n_jobs: int = 1,
) -> RandomizedSearchCV:
    """Run Random Search with a fixed budget."""
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=make_cv(n_splits=n_splits, random_state=random_state),
        random_state=random_state,
        n_jobs=n_jobs,
        refit=True,
    )
    return search.fit(x, y)


def run_bayesian_optimization_tpe(
    estimator,
    objective_fn,
    max_evals: int = 50,
    random_state: int = RANDOM_SEED,
):
    """Run Bayesian Optimization using Hyperopt TPE.

    The caller should provide an objective function compatible with hyperopt.
    Hyperopt is optional at runtime and listed in requirements.txt.
    """
    try:
        from hyperopt import Trials, fmin, tpe
    except ImportError:
        warnings.warn(
            "hyperopt is required for TPE optimization. Install it with "
            "`pip install hyperopt`. Skipping Bayesian Optimization/TPE.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None, None

    trials = Trials()
    result = fmin(
        fn=objective_fn,
        space=getattr(estimator, "search_space", {}),
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        rstate=np.random.default_rng(random_state),
    )
    return result, trials


def run_skopt_bayesian_search(
    estimator,
    search_spaces,
    x,
    y,
    n_iter: int = 50,
    scoring: str = "f1",
    n_splits: int = 5,
    random_state: int = RANDOM_SEED,
):
    """Run Bayesian optimization using scikit-optimize if available."""
    try:
        from skopt import BayesSearchCV
    except ImportError:
        warnings.warn(
            "scikit-optimize is required for BayesSearchCV. Install it with "
            "`pip install scikit-optimize`. Skipping Bayesian search.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    search = BayesSearchCV(
        estimator=estimator,
        search_spaces=search_spaces,
        n_iter=n_iter,
        scoring=scoring,
        cv=make_cv(n_splits=n_splits, random_state=random_state),
        random_state=random_state,
        n_jobs=1,
        refit=True,
    )
    return search.fit(x, y)


def run_cma_es(objective_fn, initial_params, sigma: float = 0.5, max_iter: int = 50):
    """Run CMA-ES for continuous search spaces.

    The objective function should return a loss to minimize. Encoding from a
    vector to model hyperparameters must be handled by the caller.
    """
    try:
        import cma
    except ImportError as exc:
        raise ImportError("cma is required for CMA-ES. Install it with `pip install cma`.") from exc

    options = {"maxiter": max_iter, "seed": RANDOM_SEED}
    return cma.fmin(objective_fn, initial_params, sigma, options=options)


def run_hyperband(objective_fn, config_space=None, max_epochs: int = 81):
    """Run Hyperband using Optuna's Hyperband pruner.

    The objective function should accept an Optuna trial and report intermediate
    values when possible. This function returns a completed Optuna study.
    """
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "optuna is required for Hyperband. Install it with `pip install optuna`."
        ) from exc

    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    pruner = optuna.pruners.HyperbandPruner(max_resource=max_epochs)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective_fn, n_trials=50)
    return study


if __name__ == "__main__":
    print("HPO helpers are ready. Configure dataset-specific estimators before running searches.")
