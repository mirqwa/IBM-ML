import warnings

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")


def plot_correlation_matrix(df: pd.DataFrame) -> None:
    corr_matrix = df.corr()
    sn.heatmap(corr_matrix, annot=True)
    plt.show()


def explore_data(df: pd.DataFrame) -> None:
    # EDA
    print("The number of rows and columns:", df.shape)
    print("Checking the number of columns")
    print(df.columns)
    print("The data types")
    print(df.dtypes)
    print("Checking the null values")
    print(df.isna().sum())
    print("Checking the unique counts")
    print(df.nunique())


def fill_missing_values(df: pd.DataFrame) -> None:
    df["Age"].fillna(df["Age"].mean(), inplace=True)
    df["Fare"].fillna(df["Fare"].mean(), inplace=True)
    df["Cabin"].fillna("", inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)


def convert_categorical_to_numerical(df: pd.DataFrame) -> None:
    df["Sex"].replace(to_replace=["male", "female"], value=[0, 1], inplace=True)
    df["Embarked"].replace(to_replace=["C", "Q", "S"], value=[0, 1, 2], inplace=True)


def pre_process_data(df: pd.DataFrame) -> None:
    fill_missing_values(df)
    convert_categorical_to_numerical(df)


def get_columns_to_use_in_training(df: pd.DataFrame) -> list:
    corr_df = df.corr()
    survided_corr = corr_df["Survived"]
    return survided_corr[survided_corr.abs() > 0.01].keys().tolist()


train_data = pd.read_csv("data/titanic/train.csv")
test_data = pd.read_csv("data/titanic/test.csv")

# Data exploration before preprocessing
explore_data(train_data)
explore_data(test_data)
plot_correlation_matrix(train_data)

# Data preprocessing
pre_process_data(train_data)
pre_process_data(test_data)

# Data exploration after preprocessing
print("Data exploration after pre-processing")
explore_data(train_data)
explore_data(test_data)
plot_correlation_matrix(train_data)

# Training the model
columns_to_use_in_training = get_columns_to_use_in_training(train_data)
train_data = train_data[columns_to_use_in_training]


mlp = MLPClassifier(hidden_layer_sizes=(5, 2), activation="logistic")
