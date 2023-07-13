# import traceback

# from typing import Tuple
import numpy as np
from numpy import ndarray
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import sklearn
import sklearn.pipeline
from sklearn.base import BaseEstimator
# from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report


def check_model_learning_CV(
    n_splits: int,
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.DataFrame,
    preprocess_pipeline: sklearn.pipeline.Pipeline,
    label_encoder: sklearn.preprocessing.LabelEncoder,
    random_state: int = None
) -> tuple[ndarray, ndarray, StratifiedKFold]:
    """Perform cross validation on a given model to analyze the degree of overfitting and underfitting. 
    This is achieved by evaluating the macro average f1-score on the training and test sets for each fold.
    The function also returns the used StratifiedKFold object for potential further uses.
    NOTE: moved inside a function, since transformations and training should not be puplic available
    NOTE: ignore warnings of ill-defined F-score, KFold splits might cause labels in y_true might not appear in y_pred

    Args:
        n_splits (int): The number of folds in the StratifiedKFold cross-validator.
        model (BaseEstimator): The sklearn model to be evaluated.
        X (pd.DataFrame): The input data to be used for model training and evaluation.
        y (pd.DataFrame): The target output data to be used for model training and evaluation.
        preprocess_pipeline (sklearn.pipeline.Pipeline): The preprocessing pipeline to be applied on the input data before fitting the model.
        label_encoder (sklearn.preprocessing.LabelEncoder): Fitted label encoder for target variable.
        random_state(int): Controls the shuffling applied to the data before applying the split. 
            Pass an int for reproducible output across multiple function calls.
            Default is None.

    Returns:
        performance_train (ndarray): The macro average f1-score of the model on the training set for each fold.
        performance_test (ndarray): The macro average f1-score of the model on the test set for each fold.
        kf (StratifiedKFold): The StratifiedKFold cross-validator used for cross validation.
    """

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    report_train = []
    report_test = []
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # preprocess given split
        X_train_tranformed = preprocess_pipeline.fit_transform(X_train)
        X_test_transformed = preprocess_pipeline.transform(X_test)
        y_train_transfomed = label_encoder.fit_transform(y_train)
        y_test_transformed = label_encoder.transform(y_test)

        # train model within given split
        model.fit(X_train_tranformed, y_train_transfomed)

        y_train_pred = model.predict(X_train_tranformed)
        y_test_pred = model.predict(X_test_transformed)

        # get performance results for given split
        report_train.append(classification_report(y_train_transfomed, y_train_pred, output_dict=True)['macro avg']['f1-score'])
        report_test.append(classification_report(y_test_transformed, y_test_pred, output_dict=True)['macro avg']['f1-score'])

    performance_train = np.array(report_train)
    performance_test = np.array(report_test)

    return performance_train, performance_test, kf