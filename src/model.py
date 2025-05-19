import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from mlflow.models import infer_signature
from mlflow.data import from_pandas




def logistic_regression_model(X_train, y_train, c=500, max_iter=100):
    """
    Trains and logs a logistic regression model using MLflow.

    Args:
        X_train:
        y_train:
        c:
        max_iter

    Returns:
        model:
        params:
    """

    model = LogisticRegression(C=c, max_iter=max_iter)
    model.fit(X_train, y_train)

    signature = infer_signature(X_train, y_train)
    mlflow.sklearn.log_model(model, 'Model', signature=signature, input_example=X_train)

    dataset = from_pandas(X_train, name="Bank Customer Data Testing")
    mlflow.log_input(dataset, context="training")

    params = {'c': c, 'max_iter': max_iter}

    return model, params





def random_forest_model(X_train, y_train, n_estimators=100, max_depth=10):
    """
    Trains and logs a random forest model using MLflow.
    
    Args:
        X_train:
        y_train:
        n_estimators:
        max_depth:

    Returns:
        model:
        params:
    """
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    signature = infer_signature(X_train, y_train)
    mlflow.sklearn.log_model(model, 'Model', signature=signature, input_example=X_train)

    dataset = from_pandas(X_train, name="Bank Customer Data Testing")
    mlflow.log_input(dataset, context="training")

    params = {'n_estimators': n_estimators, 'max_depth': max_depth}
    
    return model, params





def gradient_boosting_model(X_train, y_train, n_estimators=100, max_depth=3):
    """
    Trains and logs a gradient boosting model using MLflow.
    
    Args:
        X_train:
        y_train:
        n_estimators:
        max_depth:

    Returns:
        model:
        params:
    """

    model = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)

    signature = infer_signature(X_train, y_train)
    mlflow.sklearn.log_model(model, 'Model', signature=signature, input_example=X_train)

    dataset = from_pandas(X_train, name="Bank Customer Data Testing")
    mlflow.log_input(dataset, context="training")

    params = {'n_estimators': n_estimators, 'max_depth': max_depth}
    return model, params




def train(X_train, y_train, base_model, params):

    """
    Selects and trains one of the specified models based on input.
    
    Args:
        X_train:
        y_train:
        base_model:
        params:

    Returns:
        None
    """

    if base_model == 'LogisticRegression':
        return logistic_regression_model(X_train, y_train, **params)
    
    elif base_model == 'RandomForest':
        return random_forest_model(X_train, y_train, **params)
    
    elif base_model == 'GradientBoosting':
        return gradient_boosting_model(X_train, y_train, **params)
    
    else:
        raise ValueError(f"Unknown model type: {base_model}")
