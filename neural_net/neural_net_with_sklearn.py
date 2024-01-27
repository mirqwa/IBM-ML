import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier


def explore_data(df: pd.DataFrame) -> None:
    # EDA
    print("The number of rows and columns:", df.shape)
    print("Checking the number of columns")
    print(df.columns)
    print("Checking the null values")
    print(df.isna().sum())
    print("Checking the unique counts")
    print(df.nunique())


train_data = pd.read_csv("data/titanic/train.csv")
test_data = pd.read_csv("data/titanic/test.csv")

print("Exploring the training data>>>>>>>>>>>>>>>")
explore_data(train_data)

print("Exploring the test data>>>>>>>>>>>>>>>")
explore_data(test_data)


mlp = MLPClassifier(hidden_layer_sizes=(5, 2), activation="logistic")
