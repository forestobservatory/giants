from sklearn import datasets, metrics

from giants import model

# load the test classification and regression datasets
xc, yc = datasets.load_iris(return_X_y=True)
xr, yr = datasets.load_diabetes(return_X_y=True)

# set up the model tuning objects
ctuner = model.Tuner(xc, yc)
rtuner = model.Tuner(xr, yr)


def test_list_scorers_classification():
    scorers = model.list_scorers(method="classification")
    assert "accuracy" in scorers


def test_list_scorers_regression():
    scorers = model.list_scorers(method="regression")
    assert "neg_mean_squared_error" in scorers


def test_list_scorers_clustering():
    scorers = model.list_scorers(method="clustering")
    assert "adjusted_rand_score" in scorers


def test_list_scorers_all():
    scorers = model.list_scorers(method=None)
    assert "r2" in list(scorers.keys())


def test_AdaBoostClassifier():
    param_grid = {"n_estimators": (25, 50), "learning_rate": (0.1, 0.5)}
    ctuner.AdaBoostClassifier(param_grid=param_grid)
    ypred = ctuner.best_estimator.predict(xc)
    accuracy = metrics.accuracy_score(yc, ypred)
    assert accuracy > 0.0
    assert accuracy <= 1.0


def test_AdaBoostRegressor():
    param_grid = {"n_estimators": (25, 50), "learning_rate": (0.1, 0.5)}
    rtuner.AdaBoostRegressor(param_grid=param_grid)
    ypred = rtuner.best_estimator.predict(xr)
    rsquared = metrics.r2_score(yr, ypred)
    assert rsquared > 0.0
    assert rsquared <= 1.0


def test_DecisionTreeClassifier():
    param_grid = {"max_features": ("sqrt", None), "max_depth": (2, None)}
    ctuner.DecisionTreeClassifier(param_grid=param_grid)
    ypred = ctuner.best_estimator.predict(xc)
    accuracy = metrics.accuracy_score(yc, ypred)
    assert accuracy > 0.0
    assert accuracy <= 1.0


def test_GradientBoostingClassifier():
    param_grid = {"n_estimators": (10, 50), "learning_rate": (0.01, 0.1)}
    ctuner.GradientBoostingClassifier(param_grid=param_grid)
    ypred = ctuner.best_estimator.predict(xc)
    accuracy = metrics.accuracy_score(yc, ypred)
    assert accuracy > 0.0
    assert accuracy <= 1.0


def test_GradientBoostingRegressor():
    param_grid = {"n_estimators": (10, 50), "learning_rate": (0.01, 0.1)}
    rtuner.GradientBoostingRegressor(param_grid=param_grid)
    ypred = rtuner.best_estimator.predict(xr)
    rsquared = metrics.r2_score(yr, ypred)
    assert rsquared > 0.0
    assert rsquared <= 1.0


def test_LinearRegression():
    param_grid = {"normalize": (True, False), "fit_intercept": (True, False)}
    rtuner.LinearRegression(param_grid=param_grid)
    ypred = rtuner.best_estimator.predict(xr)
    rsquared = metrics.r2_score(yr, ypred)
    assert rsquared > 0.0
    assert rsquared <= 1.0


def test_LinearSVC():
    param_grid = {"C": (1e-1, 1e1), "tol": (1e-3, 1e-4)}
    ctuner.LinearSVC(param_grid=param_grid)
    ypred = ctuner.best_estimator.predict(xc)
    accuracy = metrics.accuracy_score(yc, ypred)
    assert accuracy > 0.0
    assert accuracy <= 1.0


def test_LinearSVR():
    param_grid = {"C": (1e-1, 1e1), "tol": (1e-3, 1e-4)}
    rtuner.LinearSVR(param_grid=param_grid)
    ypred = rtuner.best_estimator.predict(xr)
    rsquared = metrics.r2_score(yr, ypred)
    assert rsquared > 0.0
    assert rsquared <= 1.0


def test_LogisticRegression():
    param_grid = {"C": (1e-1, 1e1), "tol": (1e-3, 1e-4)}
    rtuner.LogisticRegression(param_grid=param_grid)
    ypred = rtuner.best_estimator.predict(xr)
    rsquared = metrics.r2_score(yr, ypred)
    assert rsquared > 0.0
    assert rsquared <= 1.0


def test_RandomForestClassifier():
    param_grid = {"n_estimators": (10, 50), "max_depth": (1, 10)}
    ctuner.RandomForestClassifier(param_grid=param_grid)
    ypred = ctuner.best_estimator.predict(xc)
    accuracy = metrics.accuracy_score(yc, ypred)
    assert accuracy > 0.0
    assert accuracy <= 1.0


def test_RandomForestRegressor():
    param_grid = {"n_estimators": (10, 50), "max_depth": (1, 10)}
    rtuner.RandomForestRegressor(param_grid=param_grid)
    ypred = rtuner.best_estimator.predict(xr)
    rsquared = metrics.r2_score(yr, ypred)
    assert rsquared > 0.0
    assert rsquared <= 1.0


def test_SVC():
    param_grid = {"C": (1e-1, 1e1), "tol": (1e-3, 1e-4)}
    ctuner.SVC(param_grid=param_grid)
    ypred = ctuner.best_estimator.predict(xc)
    accuracy = metrics.accuracy_score(yc, ypred)
    assert accuracy > 0.0
    assert accuracy <= 1.0


def test_SVR():
    param_grid = {"C": (1e-1, 1e1), "tol": (1e-3, 1e-4)}
    rtuner.SVR(param_grid=param_grid)
    ypred = rtuner.best_estimator.predict(xr)
    rsquared = metrics.r2_score(yr, ypred)
    assert rsquared > 0.0
    assert rsquared <= 1.0


def test_classification_scorers():
    param_grid = {"max_features": ("sqrt", None), "max_depth": (2, None)}
    scorers = model.list_scorers(method="classification")
    for scorer in scorers:
        ctuner.DecisionTreeClassifier(scorer=scorer, param_grid=param_grid)
        ypred = ctuner.best_estimator.predict(xc)
        accuracy = metrics.accuracy_score(yc, ypred)
        assert accuracy > 0.0
        assert accuracy <= 1.0


def test_regression_scorers():
    param_grid = {"C": (1e-1, 1e1), "tol": (1e-3, 1e-4)}
    scorers = model.list_scorers(method="regression")
    for scorer in scorers:
        rtuner.LogisticRegression(scorer=scorer, param_grid=param_grid)
        ypred = rtuner.best_estimator.predict(xr)
        rsquared = metrics.r2_score(yr, ypred)
        assert rsquared > 0.0
        assert rsquared <= 1.0
