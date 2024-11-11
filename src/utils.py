# import traceback

from typing import Dict, List, Optional, Tuple, Union

import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import sklearn
import sklearn.pipeline
from numpy import ndarray
from sklearn.base import BaseEstimator
# from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report, fbeta_score
from sklearn.model_selection import StratifiedKFold


def df_column_summary(df: pd.DataFrame = None, numeric_agg: bool = False) -> pd.DataFrame:
    """
    This function calculates summary statistics for each column in a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        numeric_agg (bool): Whether to include numeric aggregation for numeric columns in the result.

    Returns:
        pd.DataFrame: A DataFrame containing the summary statistics for each column.
    """

    # Create a new DataFrame to store results
    # general agg cols
    general_agg_cols = [
        'col_name', 'col_dtype',
        'null_present', 'n_non_nulls', 'n_nulls', 'n_nulls_pct',
        'n_unique_values', 'unique_values', 'unique_values_normalized'
    ]
    # numeric agg cols
    numeric_agg_cols = ['min_value', 'max_value', 'median_no_na', 'average_no_na', 'average_non_zero']
    result_cols = general_agg_cols + numeric_agg_cols if numeric_agg is True else general_agg_cols

    # List to hold each row's data as a dictionary
    rows = []
    # Loop through each column in the DataFrame
    for column in df.columns:
        
        # General statistics: pre-calculations
        value_counts = df[column].value_counts()
        value_counts_normalized = df[column].value_counts(normalize=True)
        
        # Base dictionary with general information
        row_data = {
            'col_name': column,
            'col_dtype': df[column].dtype,
            'null_present': df[column].isnull().any(),
            'n_non_nulls': df[column].notnull().sum(),
            'n_nulls': df[column].isnull().sum(),
            'n_nulls_pct': round((100.0*df[column].isnull().sum())/df[column].size, ndigits=2),
            'n_unique_values': len(value_counts),
            'unique_values': value_counts.head(10).to_dict(),
            'unique_values_normalized': value_counts_normalized.head(10).to_dict()
        }

        # Numeric aggregations (only if numeric_agg is True and the column is numeric)
        if numeric_agg and np.issubdtype(df[column].dtype, np.number):
            row_data.update({
                'min_value': df[column].min(),
                'max_value': df[column].max(),
                'median_no_na': df[column].median(),
                'average_no_na': df[column].mean(),
                'average_non_zero': df[column][df[column] > 0].mean()
            })

        # Append the row data to the list
        rows.append(row_data)

    # Create DataFrame from the list of row data
    result_df = pd.DataFrame(rows, columns=result_cols)
        
    return result_df


def histplot_distribution_numeric_features(df: pd.DataFrame = None, hue: Optional[str] = None, figsize_width: float = 20) -> None:
    """Histplot for numeric features"""
    # plot numeric data
    cols_numeric_feat = df.select_dtypes(include='number').columns.tolist()
    if hue:
        df_numeric_feat = df[cols_numeric_feat + [hue]]
    else:
        df_numeric_feat = df[cols_numeric_feat]
    # plot distribution of all indicators
    # Plot all features individually
    n_subplots = len(cols_numeric_feat)
    fig = plt.figure(figsize=(figsize_width, math.ceil(n_subplots/4)*5))
    fig.subplots_adjust(hspace=0.35, wspace=0.25)
    for idx, col in enumerate(cols_numeric_feat):
        idx += 1
        ax = fig.add_subplot(math.ceil(n_subplots/4),4,idx)
        sns.histplot(data=df_numeric_feat, x=col, kde=True, bins=20, hue=hue)
        # Edit graph
        ax.set_title(f'Feat: {col}')
        ax.set_xlabel(f'Bins: {col}')
        ax.set_ylabel('Quantity')
        ax.tick_params(axis = 'x', rotation = 45)

    return


def boxplot_distribution_numeric_features(df: pd.DataFrame = None, hue: Optional[str] = None, figsize_width: float = 20) -> None:
    """Box plot for numeric data"""
    cols_numeric_feat = df.select_dtypes(include='number').columns.tolist()
    if hue:
        df_numeric_feat = df[cols_numeric_feat + [hue]]
    else:
        df_numeric_feat = df[cols_numeric_feat]

    # plot distribution of all indicators
    # Plot all features individually
    n_subplots = len(cols_numeric_feat)
    fig = plt.figure(figsize=(figsize_width, math.ceil(n_subplots/4)*5))
    fig.subplots_adjust(hspace=0.35, wspace=0.25)
    for idx, col in enumerate(cols_numeric_feat):
        idx += 1
        ax = fig.add_subplot(math.ceil(n_subplots/4),4,idx)
        sns.boxplot(data=df_numeric_feat, x=col, hue=hue)
        # Edit graph
        ax.set_title(f'Feat: {col}')
        ax.set_xlabel('Value')
        ax.tick_params(axis = 'x', rotation = 45)
    
    return


def plot_distribution_numeric_features(
        df: pd.DataFrame = None,
        hue: Optional[str] = None,
        figsize_width: int = 20,
        figsize_hight_factor: float = 1.0,
    ) -> None:
    """Histplot, Boxplot and Cumulative distribution function for numeric features"""
    # plot numeric data
    cols_numeric_feat = df.select_dtypes(include='number').columns.tolist()
    if hue:
        df_numeric_feat = df[cols_numeric_feat + [hue]]
    else:
        df_numeric_feat = df[cols_numeric_feat]
    # plot distribution of all indicators
    # Plot all features individually
    n_subplots = len(cols_numeric_feat)
    fig = plt.figure(figsize=(figsize_width, math.ceil(n_subplots * 4 * figsize_hight_factor)))
    fig.subplots_adjust(hspace=0.35, wspace=0.25)
    for idx, col in enumerate(cols_numeric_feat):
        idx += 1
        # Histogram
        # ax = fig.add_subplot(math.ceil(n_subplots/4),4,idx)
        ax = fig.add_subplot(n_subplots, 3, (3*idx-2))
        sns.histplot(data=df_numeric_feat, x=col, kde=True, bins=20, hue=hue)
        ax.set_title(f'Feat: {col}')
        ax.set_xlabel(f'Bins: {col}')
        ax.set_ylabel('Quantity')
        ax.tick_params(axis = 'x', rotation = 45)
        # Boxplot
        ax = fig.add_subplot(n_subplots, 3, (3*idx-1))
        sns.boxplot(data=df_numeric_feat, x=col, hue=hue)
        ax.set_title(f'Feat: {col}')
        ax.set_xlabel('Value')
        ax.tick_params(axis = 'x', rotation = 45)
        # Cumulative distribution function
        ax = fig.add_subplot(n_subplots, 3, (3*idx))
        sns.histplot(data=df_numeric_feat, x=col, hue=hue, bins=len(df_numeric_feat), stat="density",
             element="step", fill=False, cumulative=True, common_norm=False)
        ax.set_title(f"Cumulative distribution function: {col}")

    return


def find_outliers_tukey(x: pd.Series, threshold_multiplier: float = 1.5) -> tuple[list, list]:
    """Detects outliers based on tukey IQR below Q1-1.5(Q3-Q1) or above Q3+1.5(Q3-Q1)) for a single feature

    Args:
        x (pd.Series): pandas series of a single feature

    Returns:
        list: outlier_indices
        list: outlier_values
    """

    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1 
    floor = q1 - (threshold_multiplier * iqr)
    ceiling = q3 + (threshold_multiplier * iqr)
    outlier_indices = list(x.index[(x < floor)|(x > ceiling)])
    outlier_values = list(x[outlier_indices])

    return outlier_indices, outlier_values


def f1_score_macro(
    y_true,
    y_pred,
    *,
    labels=None,
    pos_label=1,
    average="macro",
    sample_weight=None,
    zero_division="warn",
):
    """Compute the F1 score, also known as balanced F-score or F-measure.

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
            classification_report(
            y_train_transfomed, y_train_pred, output_dict=True
            )["macro avg"]["f1-score"]
        )
        report_test.append(
            classification_report(y_test_transformed, y_test_pred, output_dict=True)[
                "macro avg"
            ]["f1-score"]
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
            weights = np.sum(conf_matrix, axis=1) / np.sum(
                conf_matrix
            )  # class distribution
        else:
            # Convert class_weights dict to array in the order of classes in the confusion matrix
            weights = np.array(
                [list(class_weights.values())[i] for i in range(conf_matrix.shape[0])]
            )
        cost = np.sum(
            weights * (cost_TP * TP + cost_FN * FN + cost_FP * FP + cost_TN * TN)
        )

    elif method == "micro":
        total_TP = np.sum(TP)
        total_FP = np.sum(FP)
        total_FN = np.sum(FN)
        total_TN = np.sum(TN)
        cost = (
            cost_TP * total_TP
            + cost_FN * total_FN
            + cost_FP * total_FP
            + cost_TN * total_TN
        )

    else:
        raise ValueError(
            f"Unsupported method: {method}. Choose from 'macro', 'weighted', or 'micro'."
        )

    return round(cost, 2)
