# import all necessary libraries here
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 1. Remove Duplicates
def remove_duplicates(data):
    """
    Remove duplicate rows from the dataset.
    :param data: pandas DataFrame
    :return: pandas DataFrame
    """
    # Removing duplicate rows
    data.drop_duplicates(data)

    pass

# 2. Impute and Drop Missing Values
def impute_drop_missing_values(data, strategy='mean'):
    """
    Fill/drop missing values in the dataset.
    :param data: pandas DataFrame
    :param strategy: str, imputation method ('mean', 'median', 'mode')
    :return: pandas DataFrame
    """
    # Dropping numerical columns that have 50% or more missing values
    data.drop(columns=['j','x','z'],inplace=True)

    # Dropping rows in columns that have a binary or true/false categorization
    data.dropna(subset=['target','i','{'],inplace=True)

    # Filling missing values with "Unknown" in non-numerical columns with 35% or more missing values
    # These categorical columns seemed of great importance, and the columns or rows weren't dropped to avoid
    # losing a lot of data
    data['d'].fillna("Unknown", inplace=True)
    data['m'].fillna("Unknown", inplace=True)

    # Filling missing values in numerical columns that follow a normal distribution with "mean"
    data['o'].fillna(data['o'].mean(), inplace=True)
    data['q'].fillna(data['q'].mean(), inplace=True)
    data['u'].fillna(data['u'].mean(), inplace=True)
    data['w'].fillna(data['w'].mean(), inplace=True)
    
    # Filling missing values in numerical columns that don't follow a normal distribution with "median"
    data['h'].fillna(data['h'].median(), inplace=True)
    data['n'].fillna(data['n'].median(), inplace=True)
    data['t'].fillna(data['t'].median(), inplace=True)
    data['v'].fillna(data['t'].median(), inplace=True)

    # Filling missing values in non-numerical columns with "mode"
    data['a'].fillna(data['a'].mode()[0], inplace=True)

    pass

# 3. Identify and Remove Outliers
def remove_outliers(data, threshold=3):
    """
    Identify and remove outliers from dataset after imputation.
    :param data: pandas DataFrame
    :return: pandas DataFrame
    """
    # Identifying outliers using z-score method
    z_scores = np.abs(stats.zscore(data.select_dtypes(include=[np.number])))
    outliers = z_scores > 3
    
    # Removing outliers from DataFrame by having as threshold 3 standard deviations
    data = data[(z_scores < threshold).all(axis=1)]

    return data

# 4. Normalize Numerical Data
def normalize_data(data,method='minmax'):
    """
    Apply normalization to numerical features.
    :param data: pandas DataFrame
    :param method: str, normalization method ('minmax' (default) or 'standard')
    """
    # Normalize numerical data using Min-Max scaling
    scaler = MinMaxScaler()
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

    pass

# 5. Remove Redundant Features   
def remove_redundant_features(data, threshold=0.9):
    """
    Remove redundant or duplicate columns.
    :param data: pandas DataFrame
    :param threshold: float, correlation threshold
    :return: pandas DataFrame
    """
    # Remove redundant features based on the correlation threshold (HINT: you can use the corr() method)
    # I used ChatGPT to help me remove the redundant features from the dataset as I was having
    # a bit of an issue understanding how to work with the diagonal matrix
    columns_to_drop = set()
    correlation_matrix = data.corr(numeric_only=True)
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                colname = correlation_matrix.columns[i]
                columns_to_drop.add(colname)
    data.drop(columns=columns_to_drop, inplace=True)

    pass

# ---------------------------------------------------

def simple_model(input_data, split_data=True, scale_data=False, print_report=False):
    """
    A simple logistic regression model for target classification.
    Parameters:
    input_data (pd.DataFrame): The input data containing features and the target variable 'target' (assume 'target' is the first column).
    split_data (bool): Whether to split the data into training and testing sets. Default is True.
    scale_data (bool): Whether to scale the features using StandardScaler. Default is False.
    print_report (bool): Whether to print the classification report. Default is False.
    Returns:
    None
    The function performs the following steps:
    1. Removes columns with missing data.
    2. Splits the input data into features and target.
    3. Encodes categorical features using one-hot encoding.
    4. Splits the data into training and testing sets (if split_data is True).
    5. Scales the features using StandardScaler (if scale_data is True).
    6. Instantiates and fits a logistic regression model.
    7. Makes predictions on the test set.
    8. Evaluates the model using accuracy score and classification report.
    9. Prints the accuracy and classification report (if print_report is True).
    """

    # if there's any missing data, remove the columns
    input_data.dropna(inplace=True)

    # split the data into features and target
    target = input_data.copy()[input_data.columns[0]]
    features = input_data.copy()[input_data.columns[1:]]

    # if the column is not numeric, encode it (one-hot)
    for col in features.columns:
        if features[col].dtype == 'object':
            features = pd.concat([features, pd.get_dummies(features[col], prefix=col)], axis=1)
            features.drop(col, axis=1, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, stratify=target, random_state=42)

    if scale_data:
        # scale the data
        X_train = normalize_data(X_train)
        X_test = normalize_data(X_test)
        
    # instantiate and fit the model
    log_reg = LogisticRegression(random_state=42, max_iter=100, solver='liblinear', penalty='l2', C=1.0)
    log_reg.fit(X_train, y_train)

    # make predictions and evaluate the model
    y_pred = log_reg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f'Accuracy: {accuracy}')
    
    # if specified, print the classification report
    if print_report:
        print('Classification Report:')
        print(report)
        print('Read more about the classification report: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html and https://www.nb-data.com/p/breaking-down-the-classification')
    
    return None