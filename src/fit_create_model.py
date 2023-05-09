# linear regression model


# DVC init one folder hidden .dvc and dvc ignore file

import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

predictor = LinearRegression(n_jobs=-1)


def fit_data(train_input,train_output):
    predictor.fit(X=train_input, y=train_output)


def prepare_pkl(filename):
    # save model
    pickle.dump(predictor, open(filename, "wb"))


def load_from_csv(file_name):
    df = pd.read_csv(file_name,encoding='utf-8')
    df["input"] = df[["a", "b", "c"]].loc[0:, ].values.tolist()
    return df


if __name__ == "__main__":
    model_filename = "model/linear_regression.pkl"
    test_data_file_name = "data/text.csv"
    df = load_from_csv(test_data_file_name)
    fit_data(df["input"].to_list(),df["output"])
    prepare_pkl(model_filename)
