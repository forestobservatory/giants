"""Default configuration options for model hyperparameter searches."""
import multiprocessing as mp
from numbers import Number
from typing import Dict, Iterable, List, Union

import numpy as np
from sklearn import ensemble, linear_model, metrics, model_selection, svm, tree
from sklearn.model_selection import BaseCrossValidator, ParameterGrid
from sklearn.model_selection._search import BaseSearchCV


class TypeConfig(object):
    """Stores a series of python type hints for model-specific keywords."""

    Array = np.ndarray
    BaseCrossValidator = Union[int, BaseCrossValidator, Iterable]
    BaseSearchCV = BaseSearchCV
    ErrorScore = Union[str, Number]
    ScorerList = Union[List, Dict]
    Number = Number
    ParameterGrid = Union[Dict, ParameterGrid]


class ModelConfig(object):
    """Stores default model tuning parameters"""

    ClassificationScorer = "neg_log_loss"
    CVClassification = model_selection.StratifiedKFold
    CVRegression = model_selection.KFold
    ErrorScore = 0
    NumberOfSplits = 4
    NumberOfJobs = mp.cpu_count()
    Optimizer = model_selection.GridSearchCV
    RefitModels = True
    RegressionScorer = "neg_root_mean_squared_error"
    ReturnTrainScore = False
    Verbosity = 1


class ModelEstimatorConfig(object):
    """Stores sklearn model estimators"""

    AdaBoostClassifier = ensemble.AdaBoostClassifier
    AdaBoostRegressor = ensemble.AdaBoostRegressor
    DecisionTreeClassifier = tree.DecisionTreeClassifier
    GradientBoostingClassifier = ensemble.GradientBoostingClassifier
    GradientBoostingRegressor = ensemble.GradientBoostingRegressor
    LinearRegression = linear_model.LinearRegression
    LinearSVC = svm.LinearSVC
    LinearSVR = svm.LinearSVR
    LogisticRegression = linear_model.LogisticRegression
    RandomForestClassifier = ensemble.RandomForestClassifier
    RandomForestRegressor = ensemble.RandomForestRegressor
    SVC = svm.SVC
    SVR = svm.SVR


class ParamGridConfig(object):
    """Stores the default grid search parameters to explore for each model."""

    AdaBoostClassifier = {"n_estimators": (25, 50, 75, 100), "learning_rate": (0.1, 0.5, 1.0)}
    AdaBoostRegressor = {
        "n_estimators": (25, 50, 75, 100),
        "learning_rate": (0.1, 0.5, 1.0),
        "loss": ("linear", "exponential", "square"),
    }
    DecisionTreeClassifier = {
        "criterion": ("gini", "entropy"),
        "splitter": ("best", "random"),
        "max_features": ("sqrt", "log2", None),
        "max_depth": (2, 5, 10, None),
        "min_samples_split": (2, 0.01, 0.1),
    }
    GradientBoostingClassifier = {
        "n_estimators": (10, 100, 200),
        "learning_rate": (0.01, 0.1, 0.5),
        "max_features": ("sqrt", "log2", None),
        "max_depth": (1, 10, None),
        "min_samples_split": (2, 0.1, 0.01),
    }
    GradientBoostingRegressor = {
        "n_estimators": (10, 100, 200),
        "learning_rate": (0.01, 0.1, 0.5),
        "max_features": ("sqrt", "log2", None),
        "max_depth": (1, 10, None),
        "min_samples_split": (2, 0.1, 0.01),
    }
    LinearRegression = {"normalize": (True, False), "fit_intercept": (True, False)}
    LinearSVC = {
        "C": (1e-2, 1e-1, 1e0, 1e1),
        "loss": ("hinge", "squared_hinge"),
        "tol": (1e-3, 1e-4, 1e-5),
        "fit_intercept": (True, False),
        "class_weight": (None, "balanced"),
    }
    LinearSVR = {
        "C": (1e-2, 1e-1, 1e0, 1e1),
        "loss": ("epsilon_insensitive", "squared_epsilon_insensitive"),
        "epsilon": (0, 0.01, 0.1),
        "dual": (False),
        "tol": (1e-3, 1e-4, 1e-5),
        "fit_intercept": (True, False),
    }
    LogisticRegression = {"C": (1e-2, 1e-1, 1e0, 1e1), "tol": (1e-3, 1e-4, 1e-5), "fit_intercept": (True, False)}
    RandomForestClassifier = {
        "criterion": ("gini", "entropy"),
        "n_estimators": (10, 100, 200),
        "max_features": ("sqrt", "log2", None),
        "max_depth": (1, 10, None),
        "min_samples_split": (2, 0.1, 0.01),
    }
    RandomForestRegressor = {
        "n_estimators": (10, 100, 200),
        "max_features": ("sqrt", "log2", None),
        "max_depth": (1, 10, None),
        "min_samples_split": (2, 0.1, 0.01),
    }
    SVC = {
        "C": (1e-3, 1e-2, 1e-1, 1e0, 1e1),
        "kernel": ("rbf", "linear"),
        "gamma": (1e-3, 1e-4, 1e-5, 1e-6, 1e-7),
        "class_weight": (None, "balanced"),
    }
    SVR = {
        "C": (1e-2, 1e-1, 1e0, 1e1),
        "epsilon": (0.01, 0.1, 1),
        "kernel": ("rbf", "linear", "poly", "sigmoid"),
        "gamma": (1e-2, 1e-3, 1e-4),
    }


ModelDefaults = ModelConfig()
ModelEstimators = ModelEstimatorConfig()
ParamGrids = ParamGridConfig()
Types = TypeConfig()
