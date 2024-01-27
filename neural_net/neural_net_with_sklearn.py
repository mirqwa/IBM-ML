import warnings

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

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
    training_columns = survided_corr[survided_corr.abs() > 0.01].keys().tolist()
    training_columns.remove("Survived")
    return training_columns


train_data = pd.read_csv("data/titanic/train.csv")
test_data = pd.read_csv("data/titanic/test.csv")

# Data exploration before preprocessing
explore_data(train_data)
explore_data(test_data)
#plot_correlation_matrix(train_data)

# Data preprocessing
pre_process_data(train_data)
pre_process_data(test_data)

# Data exploration after preprocessing
print("Data exploration after pre-processing")
explore_data(train_data)
explore_data(test_data)
#plot_correlation_matrix(train_data)

# Splitting the training and the test data
training_columns = get_columns_to_use_in_training(train_data)
X_train, X_test, y_train, y_test = train_test_split(
    train_data[training_columns],
    train_data["Survived"],
    test_size=0.33,
    random_state=42,
)

# trainin the model
clf = MLPClassifier(hidden_layer_sizes=(5, 2), random_state=1, max_iter=300)
clf.fit(X_train, y_train)

score = clf.score(X_test, y_test)
print(score)