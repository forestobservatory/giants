# Model tuning

The `giants` package is a set of routines to make model tuning with `sklearn` easy.

The primary pattern is to load your dataset, create a model tuner, run the hyperparameter search, apply the predictions to test data, and evaluate performance.

## Classification example

```python
import giants
from sklearn import datasets, metrics, model_selection

# load test classification data and create train/test splits
x, y = datasets.load_iris(return_X_y=True)
xtrain, xtest, ytrain, ytest = model_selection.train_test_split(x, y, test_size=0.25)

# create the model tuner object
tuner = giants.model.Tuner(xtrain, ytrain)

# run a random forest hyperparameter search
tuner.RandomForestClassifier()

# evaluate model performance
ypred = tuner.best_estimator.predict(xtest)
print(metrics.classification_report(ytest, ypred))
```

The `tuner.best_estimator` object stores the sklearn model that minimized the scoring function provided. Each model has a default scoring function it uses, but the user can define this:

```python
tuner.RandomForestClassifier(scorer='balanced_accuracy')
```

You can list the available scorers:

```python
giants.model.list_scorers()
```

Pass your own hyperparameter grid to train models using pair-wise combinations of each hyperparameter.

```python
param_grid: {"n_estimators": (10, 50), "max_depth": (2, 10, None)}
tuner.RandomForestClassifier(param_grid=param_grid)
```

## Configuration

The hyperparameter grids, scoring functions, and other default configuration values are stored in these objects:

```python
giants.model.ModelDefaults
giants.model.ModelEstimators
giants.model.ParamGrids
```

The values are stored as attributes of these classes:

```python
print(giants.model.ModelDefaults.NumberOfSplits)

# 4
```

The code that constructs these configuration classes is in `giants.config`.

## Supported models

The following models are currently supported. This list could be easily extended to cover a broader range of existing `sklearn` models.

```python
AdaBoostClassifier
AdaBoostRegressor
DecisionTreeClassifier
GradientBoostingClassifier
GradientBoostingRegressor
LinearRegression
LinearSVC
LinearSVR
LogisticRegression
RandomForestClassifier
RandomForestRegressor
SVC
SVR
```
