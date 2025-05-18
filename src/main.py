"""
This module contains functions to preprocess and train the model
for bank consumer churn prediction.
"""



### Import MLflow
import mlflow
import pandas as pd
from model import train
import matplotlib.pyplot as plt
from preprocessing import preprocess
from sklearn.metrics import (
                                accuracy_score,
                                precision_score,
                                recall_score,
                                f1_score,
                                confusion_matrix,
                                ConfusionMatrixDisplay,
                            )






def main():

    ### Set the tracking URI for MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    ### Set the experiment name
    mlflow.set_experiment("Session 2 Experiment")


    ### Start a new run and leave all the main function code as part of the experiment
    with mlflow.start_run(run_name=f'run_{9}') as run:

        df = pd.read_csv("dataset/Churn_Modelling.csv")
        col_transf, X_train, X_test, y_train, y_test = preprocess(df)

        ### Log the max_iter parameter: n_estimators=100, max_depth=10, random_state=42
        mlflow.log_param("n_estimators", 150)
        mlflow.log_param("max_depth", 8)
        mlflow.log_param("random_state", 42)

        model = train(X_train, y_train)
        y_pred = model.predict(X_test)

        ### Log metrics after calculating them
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        #log the metrics
        mlflow.log_metrics({
                            "Accuracy": accuracy,
                            "precision": precision,
                            "recall": recall,
                            "f1-score": f1
                        })


        ### Log tag
        mlflow.set_tag("version", "1.3")
        mlflow.set_tag("model", "Random Forest")

        
        conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
        conf_mat_disp = ConfusionMatrixDisplay(
                                                confusion_matrix=conf_mat, display_labels=model.classes_
                                              )
        cm_plot = conf_mat_disp.plot()

        
        # Log the image as an artifact in MLflow
        artifact_path = 'plot_confusion_matrix.png'
        cm_plot.figure_.savefig(artifact_path)
        mlflow.log_artifact(artifact_path, artifact_path="artifacts")

        
        plt.show()



if __name__ == "__main__":
    main()
