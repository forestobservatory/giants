"""Tools for running sklearn model hyperparameter searches."""
from sklearn.metrics import _scorer

from giants.config import ModelDefaults, ModelEstimators, ParamGrids, Types


def list_scorers(method: str = None) -> Types.ScorerList:
    """Returns a list of viable scorers for grid search optimization.

    Args:
        method: Modeling method. Options: ["classification", "regression", "clustering"].
            If `None`, returns a dictionary with all available scorers.
            These methods are maintained manually and may become out of date as sklearn
            implements changes. Running `list_scorers()` without a method should provide
            the full list of `sklearn` scorers via their api.

    Returns:
        viable model performance scorer metrics.
    """
    all_scorers = _scorer.SCORERS

    classification_scorers = [
        "accuracy",
        "average_precision",
        "balanced_accuracy",
        "f1",
        "f1_micro",
        "f1_macro",
        "f1_weighted",
        "f1_samples",
        "jaccard",
        "neg_log_loss",
        "neg_brier_score",
        "precision",
        "recall",
        "roc_auc",
        "roc_auc_ovo",
        "roc_auc_ovr",
        "roc_auc_ovo_weighted",
        "roc_auc_ovr_weighted",
        "top_k_accuracy",
    ]
    regression_scorers = [
        "explained_variance",
        "max_error",
        "neg_mean_absolute_error",
        "neg_mean_poisson_deviance",
        "neg_mean_gamma_deviance",
        "neg_mean_absolute_percentage_error",
        "neg_mean_squared_error",
        "neg_mean_squared_log_error",
        "neg_median_absolute_error",
        "neg_root_mean_squared_error",
        "r2",
    ]
    clustering_scorers = [
        "adjusted_mutual_info_score",
        "adjusted_rand_score",
        "completeness_score",
        "fowlkes_mallows_score",
        "homogeneity_score",
        "mutual_info_score",
        "normalized_mutual_info_score",
        "rand_score",
        "v_measure_score",
    ]

    if method is None:
        return all_scorers

    else:
        if method.lower() == "classification":
            return classification_scorers
        elif method.lower() == "regression":
            return regression_scorers
        elif method.lower() == "clustering":
            return clustering_scorers
        else:
            return all_scorers


class Tuner(object):
    """A class for performing hyperparameter searches using `sklearn` models."""

    x = None
    y = None
    optimizer = None
    param_grid = None
    n_splits = None
    scorer = None
    fit_params = None
    n_jobs = None
    refit = None
    verbose = None
    error_score = None
    return_train_score = None
    cv = None
    cv_results = None
    best_estimator = None
    best_score = None
    best_params = None
    best_index = None
    n_splits = None
    gs = None

    def __init__(
        self,
        x: Types.Array,
        y: Types.Array,
        n_splits: int = ModelDefaults.NumberOfSplits,
        n_jobs: int = ModelDefaults.NumberOfJobs,
        refit: bool = ModelDefaults.RefitModels,
        verbose: int = ModelDefaults.Verbosity,
        error_score: Types.ErrorScore = ModelDefaults.ErrorScore,
        return_train_score: bool = ModelDefaults.ReturnTrainScore,
    ) -> object:
        """Performs hyperparameter searches to optimize model performance using `sklearn`.

        Args:
            x : {array-like, sparse matrix} of shape (n_samples, n_features)
                The input samples. `sklearn` will convert this to a `dtype=np.float32`
                array or to a sparse `csr_matrix`.
            y: array-like of shape (n_samples,)
                Target values (strings or integers in classification, real numbers in regression)
                For classification, labels correspond to classes.
            n_splits: Number of train/test splits to evaluate in cross-validation.
            n_jobs: Number of simultaneous grid search processes to run.
            refit: Whether to fit a new model each time a model class is instantiated.
            verbose: Level of model training reporting logged.
            error_score: Value to assign to the score if an error occurs in estimator fitting.
                If set to ‘raise’, the error is raised. If a numeric value is given, FitFailedWarning is raised.
            return_train_score: Get insights on how different parameter settings impact overfitting/underfitting.
                But computing scores can be expensive and is not required to select parameters that yield
                the best generalization performance.

        Returns:
            a Tuner object that can perform hyperparameter searches and optimize model performance.
        """

        self.x = x
        self.y = y
        self.n_splits = n_splits
        self.n_jobs = n_jobs
        self.refit = refit
        self.verbose = verbose
        self.error_score = error_score
        self.return_train_score = return_train_score

    # function to actually run the tuning process and report outputs
    def _run_grid_search(self, estimator):
        """Helper function to run the grid search, which is called by each of the model clases."""

        gs = self.optimizer(
            estimator,
            param_grid=self.param_grid,
            scoring=self.scorer,
            n_jobs=self.n_jobs,
            cv=self.cv,
            refit=self.refit,
            verbose=self.verbose,
            error_score=self.error_score,
            return_train_score=self.return_train_score,
        )

        if self.fit_params is None:
            gs.fit(self.x, self.y)
        else:
            gs.fit(self.x, self.y, **self.fit_params)

        self.cv_results = gs.cv_results_
        self.best_estimator = gs.best_estimator_
        self.best_score = gs.best_score_
        self.best_params = gs.best_params_
        self.best_index = gs.best_index_
        self.n_splits = gs.n_splits_
        self.gs = gs

    def AdaBoostClassifier(
        self,
        param_grid: Types.ParameterGrid = ParamGrids.AdaBoostClassifier,
        optimizer: Types.BaseSearchCV = ModelDefaults.Optimizer,
        scorer: str = ModelDefaults.ClassificationScorer,
        cv: Types.BaseCrossValidator = ModelDefaults.CVClassification,
        fit_params: dict = None,
    ) -> None:
        """Run hyperparameter tuning for an AdaBoostClassifier model.

        Args:
            param_grid: Hyperparameter values to explore in grid searching.
            optimizer: Method for evaluating hyperparameter combinations.
                From the `sklearn` group of hyperparameter optimizers.
            scorer: Performance metric to optimize in model training.
                Get available options with `list_scorers()` function.
            cv: Determines the cross-validation splitting strategy. Possible inputs for cv are:
                None, to use the default 5-fold cross validation,
                integer, to specify the number of folds in a (Stratified)KFold,
                CV splitter,
                An iterable yielding (train, test) splits as arrays of indices.
            fit_params: Parameters passed to the `fit()`` method of the estimator.
        """
        self.param_grid = param_grid
        self.optimizer = optimizer
        self.scorer = scorer
        self.cv = cv(n_splits=self.n_splits)

        estimator = ModelEstimators.AdaBoostClassifier()
        self._run_grid_search(estimator)

    def AdaBoostRegressor(
        self,
        param_grid: Types.ParameterGrid = ParamGrids.AdaBoostRegressor,
        optimizer: Types.BaseSearchCV = ModelDefaults.Optimizer,
        scorer: str = ModelDefaults.RegressionScorer,
        cv: Types.BaseCrossValidator = ModelDefaults.CVRegression,
        fit_params: dict = None,
    ) -> None:
        """Run hyperparameter tuning for an AdaBoostRegression model.

        Args:
            param_grid: Hyperparameter values to explore in grid searching.
            optimizer: Method for evaluating hyperparameter combinations.
                From the `sklearn` group of hyperparameter optimizers.
            scorer: Performance metric to optimize in model training.
                Get available options with `list_scorers()` function.
            cv: Determines the cross-validation splitting strategy. Possible inputs for cv are:
                None, to use the default 5-fold cross validation,
                integer, to specify the number of folds in a (Stratified)KFold,
                CV splitter,
                An iterable yielding (train, test) splits as arrays of indices.
            fit_params: Parameters passed to the fit method of the estimator.
        """
        self.param_grid = param_grid
        self.optimizer = optimizer
        self.scorer = scorer
        self.cv = cv(n_splits=self.n_splits)

        estimator = ModelEstimators.AdaBoostRegressor()
        self._run_grid_search(estimator)

    def DecisionTreeClassifier(
        self,
        param_grid: Types.ParameterGrid = ParamGrids.DecisionTreeClassifier,
        optimizer: Types.BaseSearchCV = ModelDefaults.Optimizer,
        scorer: str = ModelDefaults.ClassificationScorer,
        cv: Types.BaseCrossValidator = ModelDefaults.CVClassification,
        fit_params: dict = None,
    ) -> None:
        """Run hyperparameter tuning for a DecisionTreeClassifier model.

        Args:
            param_grid: Hyperparameter values to explore in grid searching.
            optimizer: Method for evaluating hyperparameter combinations.
                From the `sklearn` group of hyperparameter optimizers.
            scorer: Performance metric to optimize in model training.
                Get available options with `list_scorers()` function.
            cv: Determines the cross-validation splitting strategy. Possible inputs for cv are:
                None, to use the default 5-fold cross validation,
                integer, to specify the number of folds in a (Stratified)KFold,
                CV splitter,
                An iterable yielding (train, test) splits as arrays of indices.
            fit_params: Parameters passed to the `fit()` method of the estimator.
        """
        self.param_grid = param_grid
        self.optimizer = optimizer
        self.scorer = scorer
        self.cv = cv(n_splits=self.n_splits)

        estimator = ModelEstimators.DecisionTreeClassifier()
        self._run_grid_search(estimator)

    def GradientBoostingClassifier(
        self,
        param_grid: Types.ParameterGrid = ParamGrids.GradientBoostingClassifier,
        optimizer: Types.BaseSearchCV = ModelDefaults.Optimizer,
        scorer: str = ModelDefaults.ClassificationScorer,
        cv: Types.BaseCrossValidator = ModelDefaults.CVClassification,
        fit_params: dict = None,
    ) -> None:
        """Run hyperparameter tuning for a GradientBoostingClassifier model.

        Args:
            param_grid: Hyperparameter values to explore in grid searching.
            optimizer: Method for evaluating hyperparameter combinations.
                From the `sklearn` group of hyperparameter optimizers.
            scorer: Performance metric to optimize in model training.
                Get available options with `list_scorers()` function.
            cv: Determines the cross-validation splitting strategy. Possible inputs for cv are:
                None, to use the default 5-fold cross validation,
                integer, to specify the number of folds in a (Stratified)KFold,
                CV splitter,
                An iterable yielding (train, test) splits as arrays of indices.
            fit_params: Parameters passed to the fit method of the estimator.
        """
        self.param_grid = param_grid
        self.optimizer = optimizer
        self.scorer = scorer
        self.cv = cv(n_splits=self.n_splits)

        estimator = ModelEstimators.GradientBoostingClassifier()
        self._run_grid_search(estimator)

    def GradientBoostingRegressor(
        self,
        param_grid: Types.ParameterGrid = ParamGrids.GradientBoostingRegressor,
        optimizer: Types.BaseSearchCV = ModelDefaults.Optimizer,
        scorer: str = ModelDefaults.RegressionScorer,
        cv: Types.BaseCrossValidator = ModelDefaults.CVRegression,
        fit_params: dict = None,
    ) -> None:
        """Run hyperparameter tuning for a GradientBoostingRegressor model.

        Args:
            param_grid: Hyperparameter values to explore in grid searching.
            optimizer: Method for evaluating hyperparameter combinations.
                From the `sklearn` group of hyperparameter optimizers.
            scorer: Performance metric to optimize in model training.
                Get available options with `list_scorers()` function.
            cv: Determines the cross-validation splitting strategy. Possible inputs for cv are:
                None, to use the default 5-fold cross validation,
                integer, to specify the number of folds in a (Stratified)KFold,
                CV splitter,
                An iterable yielding (train, test) splits as arrays of indices.
            fit_params: Parameters passed to the fit method of the estimator.
        """
        self.param_grid = param_grid
        self.optimizer = optimizer
        self.scorer = scorer
        self.cv = cv(n_splits=self.n_splits)

        estimator = ModelEstimators.GradientBoostingRegressor()
        self._run_grid_search(estimator)

    def LinearRegression(
        self,
        param_grid: Types.ParameterGrid = ParamGrids.LinearRegression,
        optimizer: Types.BaseSearchCV = ModelDefaults.Optimizer,
        scorer: str = ModelDefaults.RegressionScorer,
        cv: Types.BaseCrossValidator = ModelDefaults.CVRegression,
        fit_params: dict = None,
    ) -> None:
        """Run hyperparameter tuning for a LinearRegression model.

        Args:
            param_grid: Hyperparameter values to explore in grid searching.
            optimizer: Method for evaluating hyperparameter combinations.
                From the `sklearn` group of hyperparameter optimizers.
            scorer: Performance metric to optimize in model training.
                Get available options with `list_scorers()` function.
            cv: Determines the cross-validation splitting strategy. Possible inputs for cv are:
                None, to use the default 5-fold cross validation,
                integer, to specify the number of folds in a (Stratified)KFold,
                CV splitter,
                An iterable yielding (train, test) splits as arrays of indices.
            fit_params: Parameters passed to the `fit()` method of the estimator.
        """
        self.param_grid = param_grid
        self.optimizer = optimizer
        self.scorer = scorer
        self.cv = cv(n_splits=self.n_splits)

        estimator = ModelEstimators.LinearRegression()
        self._run_grid_search(estimator)

    def LinearSVC(
        self,
        param_grid: Types.ParameterGrid = ParamGrids.LinearSVC,
        optimizer: Types.BaseSearchCV = ModelDefaults.Optimizer,
        scorer: str = ModelDefaults.ClassificationScorer,
        cv: Types.BaseCrossValidator = ModelDefaults.CVClassification,
        fit_params: dict = None,
    ) -> None:
        """Run hyperparameter tuning for a Linear Support Vector Machine classifier model.

        Args:
            param_grid: Hyperparameter values to explore in grid searching.
            optimizer: Method for evaluating hyperparameter combinations.
                From the `sklearn` group of hyperparameter optimizers.
            scorer: Performance metric to optimize in model training.
                Get available options with `list_scorers()` function.
            cv: Determines the cross-validation splitting strategy. Possible inputs for cv are:
                None, to use the default 5-fold cross validation,
                integer, to specify the number of folds in a (Stratified)KFold,
                CV splitter,
                An iterable yielding (train, test) splits as arrays of indices.
            fit_params: Parameters passed to the `fit()`` method of the estimator.
        """
        self.param_grid = param_grid
        self.optimizer = optimizer
        self.scorer = scorer
        self.cv = cv(n_splits=self.n_splits)

        estimator = ModelEstimators.LinearSVC()
        self._run_grid_search(estimator)

    def LinearSVR(
        self,
        param_grid: Types.ParameterGrid = ParamGrids.LinearSVR,
        optimizer: Types.BaseSearchCV = ModelDefaults.Optimizer,
        scorer: str = ModelDefaults.RegressionScorer,
        cv: Types.BaseCrossValidator = ModelDefaults.CVRegression,
        fit_params: dict = None,
    ) -> None:
        """Run hyperparameter tuning for a Linear Support Vector Machine regression model.

        Args:
            param_grid: Hyperparameter values to explore in grid searching.
            optimizer: Method for evaluating hyperparameter combinations.
                From the `sklearn` group of hyperparameter optimizers.
            scorer: Performance metric to optimize in model training.
                Get available options with `list_scorers()` function.
            cv: Determines the cross-validation splitting strategy. Possible inputs for cv are:
                None, to use the default 5-fold cross validation,
                integer, to specify the number of folds in a (Stratified)KFold,
                CV splitter,
                An iterable yielding (train, test) splits as arrays of indices.
            fit_params: Parameters passed to the fit method of the estimator.
        """
        self.param_grid = param_grid
        self.optimizer = optimizer
        self.scorer = scorer
        self.cv = cv(n_splits=self.n_splits)

        estimator = ModelEstimators.LinearSVR()
        self._run_grid_search(estimator)

    def LogisticRegression(
        self,
        param_grid: Types.ParameterGrid = ParamGrids.LogisticRegression,
        optimizer: Types.BaseSearchCV = ModelDefaults.Optimizer,
        scorer: str = ModelDefaults.RegressionScorer,
        cv: Types.BaseCrossValidator = ModelDefaults.CVRegression,
        fit_params: dict = None,
    ) -> None:
        """Run hyperparameter tuning for a LogisticRegression model.

        Args:
            param_grid: Hyperparameter values to explore in grid searching.
            optimizer: Method for evaluating hyperparameter combinations.
                From the `sklearn` group of hyperparameter optimizers.
            scorer: Performance metric to optimize in model training.
                Get available options with `list_scorers()` function.
            cv: Determines the cross-validation splitting strategy. Possible inputs for cv are:
                None, to use the default 5-fold cross validation,
                integer, to specify the number of folds in a (Stratified)KFold,
                CV splitter,
                An iterable yielding (train, test) splits as arrays of indices.
            fit_params: Parameters passed to the `fit()` method of the estimator.
        """
        self.param_grid = param_grid
        self.optimizer = optimizer
        self.scorer = scorer
        self.cv = cv(n_splits=self.n_splits)

        estimator = ModelEstimators.LogisticRegression()
        self._run_grid_search(estimator)

    def RandomForestClassifier(
        self,
        param_grid: Types.ParameterGrid = ParamGrids.RandomForestClassifier,
        optimizer: Types.BaseSearchCV = ModelDefaults.Optimizer,
        scorer: str = ModelDefaults.ClassificationScorer,
        cv: Types.BaseCrossValidator = ModelDefaults.CVClassification,
        fit_params: dict = None,
    ) -> None:
        """Run hyperparameter tuning for a RandomForestClassifier model.

        Args:
            param_grid: Hyperparameter values to explore in grid searching.
            optimizer: Method for evaluating hyperparameter combinations.
                From the `sklearn` group of hyperparameter optimizers.
            scorer: Performance metric to optimize in model training.
                Get available options with `list_scorers()` function.
            cv: Determines the cross-validation splitting strategy. Possible inputs for cv are:
                None, to use the default 5-fold cross validation,
                integer, to specify the number of folds in a (Stratified)KFold,
                CV splitter,
                An iterable yielding (train, test) splits as arrays of indices.
            fit_params: Parameters passed to the fit method of the estimator.
        """
        self.param_grid = param_grid
        self.optimizer = optimizer
        self.scorer = scorer
        self.cv = cv(n_splits=self.n_splits)

        estimator = ModelEstimators.RandomForestClassifier()
        self._run_grid_search(estimator)

    def RandomForestRegressor(
        self,
        param_grid: Types.ParameterGrid = ParamGrids.RandomForestRegressor,
        optimizer: Types.BaseSearchCV = ModelDefaults.Optimizer,
        scorer: str = ModelDefaults.RegressionScorer,
        cv: Types.BaseCrossValidator = ModelDefaults.CVRegression,
        fit_params: dict = None,
    ) -> None:
        """Run hyperparameter tuning for a RandomForestRegressor model.

        Args:
            param_grid: Hyperparameter values to explore in grid searching.
            optimizer: Method for evaluating hyperparameter combinations.
                From the `sklearn` group of hyperparameter optimizers.
            scorer: Performance metric to optimize in model training.
                Get available options with `list_scorers()` function.
            cv: Determines the cross-validation splitting strategy. Possible inputs for cv are:
                None, to use the default 5-fold cross validation,
                integer, to specify the number of folds in a (Stratified)KFold,
                CV splitter,
                An iterable yielding (train, test) splits as arrays of indices.
            fit_params: Parameters passed to the fit method of the estimator.
        """
        self.param_grid = param_grid
        self.optimizer = optimizer
        self.scorer = scorer
        self.cv = cv(n_splits=self.n_splits)

        estimator = ModelEstimators.RandomForestRegressor()
        self._run_grid_search(estimator)

    def SVC(
        self,
        param_grid: Types.ParameterGrid = ParamGrids.SVC,
        optimizer: Types.BaseSearchCV = ModelDefaults.Optimizer,
        scorer: str = ModelDefaults.ClassificationScorer,
        cv: Types.BaseCrossValidator = ModelDefaults.CVClassification,
        fit_params: dict = None,
    ) -> None:
        """Run hyperparameter tuning for a Support Vector Machine classifier (SVC) model.

        Args:
            param_grid: Hyperparameter values to explore in grid searching.
            optimizer: Method for evaluating hyperparameter combinations.
                From the `sklearn` group of hyperparameter optimizers.
            scorer: Performance metric to optimize in model training.
                Get available options with `list_scorers()` function.
            cv: Determines the cross-validation splitting strategy. Possible inputs for cv are:
                None, to use the default 5-fold cross validation,
                integer, to specify the number of folds in a (Stratified)KFold,
                CV splitter,
                An iterable yielding (train, test) splits as arrays of indices.
            fit_params: Parameters passed to the `fit()`` method of the estimator.
        """
        self.param_grid = param_grid
        self.optimizer = optimizer
        self.scorer = scorer
        self.cv = cv(n_splits=self.n_splits)

        estimator = ModelEstimators.SVC()
        self._run_grid_search(estimator)

    def SVR(
        self,
        param_grid: Types.ParameterGrid = ParamGrids.SVR,
        optimizer: Types.BaseSearchCV = ModelDefaults.Optimizer,
        scorer: str = ModelDefaults.RegressionScorer,
        cv: Types.BaseCrossValidator = ModelDefaults.CVRegression,
        fit_params: dict = None,
    ) -> None:
        """Run hyperparameter tuning for a Support Vector Machine Regression (SVR) model.

        Args:
            param_grid: Hyperparameter values to explore in grid searching.
            optimizer: Method for evaluating hyperparameter combinations.
                From the `sklearn` group of hyperparameter optimizers.
            scorer: Performance metric to optimize in model training.
                Get available options with `list_scorers()` function.
            cv: Determines the cross-validation splitting strategy. Possible inputs for cv are:
                None, to use the default 5-fold cross validation,
                integer, to specify the number of folds in a (Stratified)KFold,
                CV splitter,
                An iterable yielding (train, test) splits as arrays of indices.
            fit_params: Parameters passed to the fit method of the estimator.
        """
        self.param_grid = param_grid
        self.optimizer = optimizer
        self.scorer = scorer
        self.cv = cv(n_splits=self.n_splits)

        estimator = ModelEstimators.SVR()
        self._run_grid_search(estimator)
