import pandas as pd
from sklearn.linear_model import LinearRegression
import mlflow

predictor = LinearRegression(n_jobs=-1)

def create_experiment(experiment_name, run_name,  model, run_params=None):
    # mlflow.set_tracking_uri("http://localhost:5000") #uncomment this line if you want to use any database like sqlite as backend storage for model
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):

        if not run_params == None:
            for param in run_params:
                mlflow.log_param(param, run_params[param])

        # for metric in run_metrics:
        #     mlflow.log_metric(metric, run_metrics[metric])

        mlflow.sklearn.log_model(model, "model")

        mlflow.set_tag("tag1", "Random Forest")
        mlflow.set_tags({"tag2": "Randomized Search CV", "tag3": "Production"})

    print('Run - %s is logged to Experiment - %s' % (run_name, experiment_name))


def train_model(train_input,train_output):
    predictor.fit(X=train_input, y=train_output)


def load_from_csv(file_name):
    df = pd.read_csv(file_name,encoding='utf-8')
    df["input"] = df[["a", "b", "c"]].loc[0:, ].values.tolist()
    return df


if __name__ == "__main__":
    model_filename = "model/linear_regression.pkl"
    test_data_file_name = "data/text.csv"
    df = load_from_csv(test_data_file_name)
    train_model(df["input"].to_list(),df["output"])

    experiment_name = "linear_reg"
    run_name = "testing"
    create_experiment(experiment_name, run_name, predictor)


