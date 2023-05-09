# linear regression model
# DVC init one folder hidden .dvc and dvc ignore file

from random import randint
import pandas as pd
import dvc.api


def generate_test_data(train_set_limit,train_set_no_of_records,hyper):
    train_input_a = list()
    train_input_b = list()
    train_input_c = list()
    train_output = list()
    for i in range(train_set_no_of_records):
        a = randint(0, train_set_limit)
        b = randint(0, train_set_limit)
        c = randint(0, train_set_limit)
        op = a + (2 * b) + (3 * c)+hyper
        train_input_a.append(a)
        train_input_b.append(b)
        train_input_c.append(c)
        train_output.append(op)
    df = pd.DataFrame()
    df['a'] = train_input_a
    df['b'] = train_input_b
    df['c'] = train_input_c
    df['output'] = train_output
    store_into_csv(df)


def store_into_csv(df):
    df.to_csv("data/text.csv",index=False,encoding='utf-8')


if __name__=="__main__":
    params = dvc.api.params_show()
    generate_test_data(params['train_set_limit'],params['train_set_no_of_records'],params['hyper'])


