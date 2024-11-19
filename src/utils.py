# import traceback

from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import sklearn.pipeline
from numpy import ndarray
from sklearn.base import BaseEstimator

# from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report, fbeta_score
from sklearn.model_selection import StratifiedKFold


def f1_score_macro(
    y_true,
    y_pred,
    *,
    labels=None,
    pos_label=1,
    average="macro",
    sample_weight=None,
    zero_division="warn",
) -> ndarray | float | float | float | float:
    """Compute the F1 score, also known as balanced F-score or F-measure.
    This function sets the average parameter to "macro" by default, so the function can be more easily used
    in the context of a pipeline without having to specify the average parameter.

    The F1 score can be interpreted as a harmonic mean of the precision and
    recall, where an F1 score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F1 score are
    equal. The formula for the F1 score is:

    .. math::
        \\text{F1} = \\frac{2 * \\text{TP}}{2 * \\text{TP} + \\text{FP} + \\text{FN}}

    Where :math:`\\text{TP}` is the number of true positives, :math:`\\text{FN}` is the
    number of false negatives, and :math:`\\text{FP}` is the number of false positives.
    F1 is by default
    calculated as 0.0 when there are no true positives, false negatives, or
    false positives.
    """

    return fbeta_score(
        y_true,
        y_pred,
        beta=1,
        labels=labels,
        pos_label=pos_label,
        average=average,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )


def check_model_learning_CV(
    n_splits: int,
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.DataFrame,
    preprocess_pipeline: sklearn.pipeline.Pipeline,
    label_encoder: sklearn.preprocessing.LabelEncoder,
    random_state: int = None,
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
        report_train.append(
            classification_report(y_train_transfomed, y_train_pred, output_dict=True)["macro avg"][
                "f1-score"
            ]
        )
        report_test.append(
            classification_report(y_test_transformed, y_test_pred, output_dict=True)["macro avg"][
                "f1-score"
            ]
        )

    performance_train = np.array(report_train)
    performance_test = np.array(report_test)

    return performance_train, performance_test, kf


def calculate_weighted_cost(
    conf_matrix: np.ndarray,
    cost_matrix: List[List[float]],
    method: str = "macro",
    class_weights: Optional[Dict[int, float]] = None,
) -> float:
    """This function calculates the weighted cost for a multi-class classification model based on different averaging methods.
    Function's logic:
    Initial Setup: Extract costs for TP, FN, FP, TN from the given 2x2 cost matrix.
    Validation: If class weights are provided, check to ensure they match the classes derived from the confusion matrix.
    Calculate Basic Metrics: Compute TP, FN, FP, TN for each class based on the confusion matrix.
    Averaging:
      For macro, calculate the cost for each class independently and average the results.
      For weighted, calculate the cost for each class, then weight the results by either the provided class weights or the class distribution derived from the confusion matrix, and then average.
      For micro, aggregate TP, FN, FP, TN from all classes, then calculate the total cost.

    Args:
        conf_matrix (np.ndarray): Confusion matrix for multi-class classification.
        cost_matrix (List[List[float]]): 2x2 cost matrix in the format [[TP, FN], [FP, TN]].
        method(str): Averaging method. The averaging method to use. It can be one of {'macro', 'weighted', 'micro'}. Defaults to 'macro'.
        class_weights(Optional[Dict[int, float]]): Optional dictionary of class weights. Must be in the same order as the labels used in the confusion matrix.
            If not provided, the class distribution derived from the confusion matrix is used for the 'weighted' method. Defaults to None.

    Raises:
        ValueError: Error handeling for "method"

    Returns:
        float: The weighted cost based on the selected averaging method.

    ### Documentation ###
    Calculate Weighted Cost for Multi-Class Classification:
    - This approach integrates both statistical measures and domain-specific costs to provide a more business-oriented performance metric.
    - Key Concepts:
      - Confusion Matrix: This is a table that describes the performance of a classification model by comparing predicted and true class labels. For a multi-class classification problem with n classes, the confusion matrix will be n x n.
      - Cost Matrix: Typically used in business contexts to quantify the implications of false positives, false negatives, true positives, and true negatives. In this implementation, a 2x2 cost matrix is used in the format: [[TP, FN], [FP, TN]]
      - Averaging Methods: Three methods are used to aggregate costs across classes:
        - Macro-average: Computes the metric independently for each class and then takes the average, treating all classes equally.
        - Weighted-average: Computes the metric for each class, but when averaging, it weights the metric by the number of true instances for each class.
        - Micro-average: Aggregates contributions from all classes to compute the average metric.
    - Class Weights: Allow for assigning different importance to classes. It's a dictionary with class labels as keys and weights as values.

    Important Notes:
    - Ensure the confusion matrix reflects the actual vs. predicted classifications for the model.
    - Adjust the values in the cost matrix to represent actual business costs or penalties.
    - If using class weights, ensure they're provided in a dictionary format and match the classes in the confusion matrix.
    - Choose the averaging method that aligns with your business context.
    ### Documentation end ###

    Examples:
    >>> # Example true and predicted labels
    >>> y_true = ["cat", "dog", "bird", "cat", "bird"]
    >>> y_pred = ["cat", "dog", "cat", "cat", "bird"]
    <...>
    >>> # Specifying class labels
    >>> class_labels = np.sort(["bird", "dog", "cat"])  # order in which classes appear in the confusion matrix
    <...>
    >>> conf_matrix = confusion_matrix(y_true, y_pred, labels=class_labels) # conf_matrix = np.array([[2, 1, 0], [0, 2, 1], [1, 0, 2]])
    >>> conf_matrix = np.array([[2, 1, 0], [0, 2, 1], [1, 0, 2]])
    >>> cost_matrix = np.array([[1, -10], [-5, 1]])
    >>> class_weights_dict = {0: 0.8, 1: 0.5, 2: 0.2}
    <...>
    >>> cost_macro = calculate_weighted_cost(conf_matrix, cost_matrix, method='macro')
    >>> cost_weighted = calculate_weighted_cost(conf_matrix, cost_matrix, method='weighted', class_weights=class_weights_dict) # class_weights must be in the same order as the labels used in the confusion matrix.
    >>> cost_micro = calculate_weighted_cost(conf_matrix, cost_matrix, method='micro')
    <...>
    >>> print("Macro-averaged cost: ", cost_macro)
    >>> print("Weighted-averaged cost: ", cost_weighted)
    >>> print("Micro-averaged cost: ", cost_micro)
    """

    # Convert cost_matrix to numpy array if it isn't already
    cost_matrix = np.array(cost_matrix)

    # Extract costs
    cost_TP = cost_matrix[0, 0]
    cost_FN = cost_matrix[0, 1]
    cost_FP = cost_matrix[1, 0]
    cost_TN = cost_matrix[1, 1]

    # Check if class_weights match the classes from conf_matrix
    if class_weights:
        assert (
            len(class_weights) == conf_matrix.shape[0]
        ), "Mismatch between class weights and confusion matrix classes."

    FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix)
    FN = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
    TP = np.diag(conf_matrix)
    TN = conf_matrix.sum() - (FP + FN + TP)

    if method == "macro":
        cost = np.mean(cost_TP * TP + cost_FN * FN + cost_FP * FP + cost_TN * TN)

    elif method == "weighted":
        if not class_weights:
            weights = np.sum(conf_matrix, axis=1) / np.sum(conf_matrix)  # class distribution
        else:
            # Convert class_weights dict to array in the order of classes in the confusion matrix
            weights = np.array(
                [list(class_weights.values())[i] for i in range(conf_matrix.shape[0])]
            )
        cost = np.sum(weights * (cost_TP * TP + cost_FN * FN + cost_FP * FP + cost_TN * TN))

    elif method == "micro":
        total_TP = np.sum(TP)
        total_FP = np.sum(FP)
        total_FN = np.sum(FN)
        total_TN = np.sum(TN)
        cost = cost_TP * total_TP + cost_FN * total_FN + cost_FP * total_FP + cost_TN * total_TN

    else:
        raise ValueError(
            f"Unsupported method: {method}. Choose from 'macro', 'weighted', or 'micro'."
        )

    return round(cost, 2)
