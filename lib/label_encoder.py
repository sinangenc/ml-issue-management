import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


def fixers_binary_encoding(X_train, X_test):
    combined_column = pd.concat([X_train["fixers"], X_test["fixers"]], ignore_index=True)
    combined_column_df = combined_column.to_frame(name="fixers")

    # Create OneHotEncoder object and train it
    mlb = MultiLabelBinarizer()
    mlb.fit(combined_column_df["fixers"])

    # Transform 'fixers' column for both training and test datasets
    training_fixers_list = mlb.transform(X_train["fixers"])
    test_fixers_list = mlb.transform(X_test["fixers"])

    dev_names = mlb.classes_

    return training_fixers_list, test_fixers_list, dev_names


def labels_binary_encoding(X_train, X_test):
    combined_column = pd.concat([X_train["labels"], X_test["labels"]], ignore_index=True)
    combined_column_df = combined_column.to_frame(name="labels")

    # Create OneHotEncoder object and train it
    mlb = MultiLabelBinarizer()
    mlb.fit(combined_column_df["labels"])

    # Transform 'fixers' column for both training and test datasets
    training_labels = mlb.transform(X_train["labels"])
    test_labels = mlb.transform(X_test["labels"])

    label_names = mlb.classes_

    return training_labels, test_labels, label_names
