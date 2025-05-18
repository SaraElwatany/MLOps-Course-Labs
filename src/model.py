import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier







def train(X_train, y_train):
    """
    Train a logistic regression model.

    Args:
        X_train (pd.DataFrame): DataFrame with features
        y_train (pd.Series): Series with target

    Returns:
        LogisticRegression: trained logistic regression model
    """

    # log_reg = LogisticRegression(C=500, max_iter=100)
    # log_reg.fit(X_train, y_train)

    # ### Log the model with the input and output schema
    # # Infer signature (input and output schema)

    # # Log model
    # signature = mlflow.models.infer_signature(X_train, y_train) ### INFERING THE SIGNATURE
    # mlflow.sklearn.log_model(log_reg, 'Model', signature=signature, input_example=X_train)            

    # ### Log the data
    # dataset = mlflow.data.from_pandas(
    #                                     X_train, name="Bank Customer Data Testing"
    #                                  )
    # mlflow.log_input(dataset, context="testing")


    # return log_reg


    # Train the model
    rf_model = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
    rf_model.fit(X_train, y_train)

    # Log the model with the input and output schema
    signature = mlflow.models.infer_signature(X_train, y_train)
    mlflow.sklearn.log_model(rf_model, 'Model', signature=signature, input_example=X_train)

    # Log the data
    dataset = mlflow.data.from_pandas(X_train, name="Bank Customer Data Testing")
    mlflow.log_input(dataset, context="testing")

    return rf_model