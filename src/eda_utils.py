from typing import Dict, List, Optional, Tuple, Union

import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


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
