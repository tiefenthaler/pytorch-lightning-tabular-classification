from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from .tabular_lightning import (MulticlassTabularLightningModule,
                                TabularDataModuleClassificationPACKAGING)


def check_data_consitancy(dm: TabularDataModuleClassificationPACKAGING = None):
    tabular_data_full = pd.concat([
        dm.train_dataset.get_dataframe,
        dm.val_dataset.get_dataframe,
        dm.test_dataset.get_dataframe
    ], axis=0, ignore_index=True)
    # check column names
    assert dm.data.columns.tolist() == tabular_data_full.columns.tolist()
    # check shape of input data and processed data
    assert dm.data.shape == tabular_data_full.shape


def check_dataloader_output(
    dm: TabularDataModuleClassificationPACKAGING = None,
    out: Dict[str, torch.Tensor] = None,
):
    """Tests the output of the dataloader."""

    continuous_x = out["continuous"]
    categorical_x = out["categorical"]
    y = out["target"]

    assert isinstance(y, torch.Tensor), "y output should be a torch.Tensor"

    # check continuous features for nans and finite
    assert torch.isfinite(continuous_x).all(), f"Values for {categorical_x} should be finite"
    assert not torch.isnan(continuous_x).any(), f"Values for {categorical_x} should not be nan"
    assert continuous_x.dtype == torch.float32, f"Values for {categorical_x} should be of type float32"
    # check categorical features for nans and finite
    assert torch.isfinite(categorical_x).all(), f"Values for {categorical_x} should be finite"
    assert not torch.isnan(categorical_x).any(), f"Values for {categorical_x} should not be nan"
    assert categorical_x.dtype == torch.int64, f"Values for {categorical_x} should be of type int64"

    # check target for nans and finite
    assert torch.isfinite(y).all(), "Values for target should be finite"
    assert not torch.isnan(y).any(), "Values for target should not be nan"
    assert y.dtype == torch.int64, "Values for target should be of type int64"

    # check shape
    assert continuous_x.size(1) == dm.data[dm.continuous_cols].shape[1]
    assert categorical_x.size(1) == dm.data[dm.categorical_cols].shape[1]


def print_dataloader_output(dm: TabularDataModuleClassificationPACKAGING = None):
    """Prints the output of the dataloader."""
    num_epochs = 1
    for epoch in range(num_epochs):

        for batch_idx, dict in enumerate(dm.train_dataloader()):
            print("Batch:", batch_idx)
            if batch_idx >= 1:
                break
            for k, v in dict.items():
                print(k, v.shape)

            network_input = torch.cat((dict["continuous"], dict["categorical"]), dim=1)
            print(
                "Shape of network input:", network_input.shape,
                "Data Types Cont:", [column.dtype for column in dict["continuous"].unbind(1)],
                "Data Types Cat:", [column.dtype for column in dict["categorical"].unbind(1)],
            )
            # print("Shape of target flatten:", dict['target'].flatten().shape, "Data Types:", dict['target'].flatten().dtype)
            print("Shape of target flatten:", dict['target'].shape, "Data Types:", dict['target'].dtype)
            print("Target from current batch:", dict['target'][:5])
            print("Dataloader output from current batch, Cont:", dict["continuous"][:3])
            print("Dataloader output from current batch, Cat:", dict["categorical"][:3])


def get_embedding_size(n: int, max_size: int = 100) -> int:
    """
    Determine empirically good embedding sizes (formula taken from fastai).

    Args:
        n (int): number of classes
        max_size (int, optional): maximum embedding size. Defaults to 100.

    Returns:
        int: embedding size
    """
    if n > 2:
        return min(round(1.6 * n**0.56), max_size)
    else:
        return 1


def get_cat_feature_embedding_sizes(data: pd.DataFrame = None, categorical_cols = None) -> None:
    embedding_sizes_cat_features = {
        cat_feature: (data[cat_feature].nunique(), get_embedding_size(data[cat_feature].nunique()))
        for cat_feature in data.columns if cat_feature in categorical_cols
    }
    # add 1 to the embedding size to account for the padding token, input tensor must be within the expected range [0, num_embeddings-1]
    embedding_sizes_cat_features = {
        key: (first + 1, second) for key, (first, second) in embedding_sizes_cat_features.items()
    }

    return embedding_sizes_cat_features


def print_embbeding_input_output(dm: TabularDataModuleClassificationPACKAGING = None):
    """
    Args:
        dm: pre-processed datamodule from class TabularDataModuleClassificationPACKAGING
    """

    num_epochs = 1
    for epoch in range(num_epochs):

        for batch_idx, dict in enumerate(dm.train_dataloader()):
            print("Batch:", batch_idx)
            if batch_idx >= 1:
                break
            for k, v in dict.items():
                print(k, v.shape)

            network_input = torch.cat((dict["continuous"], dict["categorical"]), dim=1)
            print(
                "Shape of network input:", network_input.shape,
                "Data Types Cont:", [column.dtype for column in dict["continuous"].unbind(1)],
                "Data Types Cat:", [column.dtype for column in dict["categorical"].unbind(1)]
            )
            print("Shape of target flatten:", dict['target'].shape, "Data Types:", dict['target'].dtype)
            print("Dataloader output from current batch, Cont:\n", dict["continuous"][:3])
            print("Dataloader output from current batch, Cat:\n", dict["categorical"][:3])
            # print("Dataloader output from current batch, Cat Feature 0:\n", dict["categorical"][:,0])

            tabular_data_full = pd.concat(
                [dm.train_dataset.get_dataframe, dm.val_dataset.get_dataframe, dm.test_dataset.get_dataframe],
                axis=0, ignore_index=True
            )
            embedding_sizes_cat_features = get_cat_feature_embedding_sizes(tabular_data_full, categorical_cols=dm.categorical_cols)
            embedding_sizes=embedding_sizes_cat_features
            cat_embeddings = nn.ModuleDict()
            for name in embedding_sizes.keys():
                cat_embeddings[name] = nn.Embedding(
                    embedding_sizes[name][0],
                    embedding_sizes[name][1],
                )
            output_vectors = {}
            for idx, (name, emb) in enumerate(cat_embeddings.items()):
                print(name, "- Min, Max: ", dict["categorical"][:, idx].min(), dict["categorical"][:, idx].max())
                output_vectors[name] = emb(dict["categorical"][:, idx])
            embed_vector_cat = torch.cat(list(output_vectors.values()), dim=1)
            print("Shape Embeddings from current batch, Cat:", embed_vector_cat.shape)
            print("Embeddings from current batch, Cat:\n", embed_vector_cat[0])


def print_model_summary(model):
    """Prints a summary of a PyTorch model."""
    # Accessing individual parameters
    for name, param in model.named_parameters():
        print(f"Parameter name: {name}, Shape: {param.shape}")
    # Getting the total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    return


def plot_training_metrics(metrics: pd.DataFrame, **kwargs):
    """
    Plot metrics from a dataframe.
    Args:
        metrics (pd.DataFrame): metrics dataframe from lightning trainer logger
        **kwargs: additional arguments to pass to the plot function

    Returns:
        pd.DataFrame: dataframe with the metrics
    """
    aggreg_metrics = []
    agg_col = "epoch"
    for i, dfg in metrics.groupby(agg_col):
        agg = dict(dfg.mean())
        agg[agg_col] = i
        aggreg_metrics.append(agg)

    df_metrics = pd.DataFrame(aggreg_metrics)

    # Plotting
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
    # Plot 1: Loss
    df_metrics[["train_loss", "val_loss"]].plot(
        ax=axes[0], grid=True, legend=True, xlabel="Epoch", ylabel="Loss", **kwargs
    )
    # Plot 2: Accuracy
    df_metrics[["train_F1_macro_weighted", "val_F1_macro_weighted"]].plot(
        ax=axes[1], grid=True, legend=True, xlabel="Epoch", ylabel="Accuracy", **kwargs
    )

    plt.tight_layout()
    plt.show()

    return df_metrics
