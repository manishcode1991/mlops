# linear regression model


# DVC init one folder hidden .dvc and dvc ignore file
import pickle

# load model
loaded_model = pickle.load(open("model/linear_regression.pkl", "rb"))
# you can use loaded model to compute predictions


def predict(X_TEST):
    y_predicted = loaded_model.predict(X=X_TEST)
    return y_predicted


if __name__=="__main__":
    text = [[10, 20, 30]]
    print(predict(text))