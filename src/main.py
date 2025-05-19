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

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Session 2 Experiment")

    df = pd.read_csv("dataset/Churn_Modelling.csv")
    col_transf, X_train, X_test, y_train, y_test = preprocess(df)

    models = ['LogisticRegression', 'RandomForest', 'GradientBoosting']
    model_id = 0


    model_params = {
        'LogisticRegression': [
            {'c': 1, 'max_iter': 100},
            {'c': 10, 'max_iter': 150},
            {'c': 100, 'max_iter': 200},
            {'c': 500, 'max_iter': 250},
            {'c': 1000, 'max_iter': 300}
        ],

        'RandomForest': [
            {'n_estimators': 50, 'max_depth': 5},
            {'n_estimators': 100, 'max_depth': 10},
            {'n_estimators': 150, 'max_depth': 15},
            {'n_estimators': 200, 'max_depth': 20},
            {'n_estimators': 250, 'max_depth': 25}
        ],

        'GradientBoosting': [
            {'n_estimators': 50, 'max_depth': 3},
            {'n_estimators': 100, 'max_depth': 4},
            {'n_estimators': 150, 'max_depth': 5},
            {'n_estimators': 200, 'max_depth': 6},
            {'n_estimators': 250, 'max_depth': 7}
        ]
    }




    for run_id in range(10, 25):


        if run_id % 5 == 0 and run_id != 10:
            model_id += 1

        current_model = models[model_id]

        trial_id = (run_id - 10) % 5
        current_params = model_params[current_model][trial_id]



        with mlflow.start_run(run_name=f'run_{str(run_id)}', nested=True):

            try:

                for key, val in current_params.items():
                    mlflow.log_param(key, val)


                model, _ = train(X_train, y_train, current_model, current_params)
                y_pred = model.predict(X_test)



                mlflow.log_metrics({
                                    "accuracy": accuracy_score(y_test, y_pred),
                                    "precision": precision_score(y_test, y_pred),
                                    "recall": recall_score(y_test, y_pred),
                                    "f1_score": f1_score(y_test, y_pred)
                                  })


                mlflow.set_tag("version", "1.3")
                mlflow.set_tag("model", current_model)

                conf_mat = confusion_matrix(y_test, y_pred)
                disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
                disp.plot()

                artifact_path = 'plot_confusion_matrix.png'
                plt.savefig(artifact_path)
                mlflow.log_artifact(artifact_path, artifact_path="artifacts")
                plt.close()

            finally:
                mlflow.end_run()





if __name__ == "__main__":
    main()
