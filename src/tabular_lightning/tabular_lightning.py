# Data related classes for PyTorch Lightning based Tabular Prediction.

from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from imblearn.over_sampling import RandomOverSampler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (LabelEncoder, MinMaxScaler, OneHotEncoder,
                                   OrdinalEncoder, StandardScaler)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Metric

from .encoders import OrdinalEncoderExtensionUnknowns


class TabularDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame = None,
        continuous_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        target: Optional[List[Union[str, int, float]]] = None,
        task: Literal["classification", "regression"] = "classification",
    ) -> None:
        """
        This class is customized for tabular related data for the use of classification and regression. Returns the tabular data as tensor format.
        Input data must be of numeric nature and should be ordinal or label encoded. This should be covered by a related LightningDataModule.
        Besides the standard functionality of the 'Dataset' class it provides data type correction to fit the requirements of Neural Networks and for efficent use of Neural Networks.
        NOTE: The common/original intention of using a Torch Dataset Class, is to provide the output of the data as tensors for further use of pytorch
              and to enable tensor operations. For our (and most) tabular datasets we neglect the aspect of tensor operations, since we do the data transformations (e.g. using sklearn),
              which are not tensor based, within a L.LightningDataModule. The TabularDataset class is used to provide the data as tensors to the DataLoaders as a final step after data prepressing.

        Args:
            data (pd.DataFrame): Pandas DataFrame to load during training, validation, testing and prediction.
            continuous_cols (List[str], optional): A list of names of continuous columns.
            categorical_cols (List[str], optional): A list of names of categorical columns. These columns must be ordinal or label encoded beforehand.
            target (List[str], optional): A list of strings with target column name(s).
            task (str): Whether it is a classification or regression task. If classification, it returns a LongTensor as target.
        Returns:
            Corrected tabular data as tensor format.
        """
        # self.data = data
        self.task = task
        self.n_samples = data.shape[0]
        self.categorical_cols = categorical_cols
        self.continuous_cols = continuous_cols
        self.target = target

        # NOTE: input data must be ordinal or label encoded

        # target handling
        if self.target:
            self.y = data[self.target].astype(np.float32).values  # for regression task
            if self.task == "classification":
                # self.y = self.y.reshape(-1, 1).astype(np.int64) # for classification task, reshape for multi class classification (must be handled accordingly in the model)
                self.y = self.y.astype(np.int64)  # for classification task
        else:
            self.y = np.zeros((self.n_samples, 1))  # for regression task
            if self.task == "classification":
                self.y = self.y.astype(np.int64)  # for classification task

        # feature handling
        self.categorical_cols = self.categorical_cols if self.categorical_cols else []
        self.continuous_cols = self.continuous_cols if self.continuous_cols else []
        if self.continuous_cols:
            self.continuous_X = data[self.continuous_cols].astype(np.float32).values
        if self.categorical_cols:
            self.categorical_X = data[self.categorical_cols].astype(np.int64).values
            # self.categorical_X = self.categorical_X.astype(np.int64) # TODO: remove

    @property
    def get_dataframe(self) -> pd.DataFrame:
        """Creates and returns the dataset as a pandas dataframe."""
        if self.continuous_cols or self.categorical_cols:
            df = pd.DataFrame(
                dict(zip(self.continuous_cols, self.continuous_X.T))
                | dict(zip(self.categorical_cols, self.categorical_X.T))
            )
        else:
            df = pd.DataFrame()
        df[self.target] = self.y  # add target column

        return df

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return self.n_samples

    def __getitem__(self, idx: int = None) -> Dict[str, torch.Tensor]:
        """
        Generates one single sample of data of the dataset (row)
        and applies transformations to that sample if defined.
        Called iteratively based on batches and batch_size.
        Args:
            idx (int): index (between ``0`` and ``len(dataset) - 1``)

        Returns:
            Tuple[Dict[str, torch.Tensor], torch.Tensor]: x and y for model
        """

        return {
            "continuous": (torch.as_tensor(self.continuous_X[idx]) if self.continuous_cols else torch.Tensor()),
            "categorical": (torch.as_tensor(self.categorical_X[idx]) if self.categorical_cols else torch.Tensor()), #  dtype=torch.int64
            "target": torch.as_tensor(self.y[idx]) # , dtype=torch.long
        }


class TabularDataModuleClassificationPACKAGING(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        continuous_cols: List[str] = None,
        categorical_cols: List[str] = None,
        target: List[str] = None,
        oversampling: bool = False,
        task_dataset: str = "classification",
        test_size: Optional[float] = None,
        val_size: Optional[float] = None,
        batch_size: int = 64,
        batch_size_inference: Optional[int] = None,
        num_workers_train: int = 0,
        num_workers_predict: int = 0,
        kwargs_dataloader_trainvaltest: Dict = {},
        kwargs_dataloader_predict: Dict = {},
        SEED: Optional[int] = 42,
    ) -> None:
        """
        The class processes the data accordingly, so that the output meets the requirments to be further use of PyTorch/Lightning.
        A shareble, reusable class that encapsulates data loading and data preprocessing logic for classification.
        The class provides general data handeling and very specific data handeling to the 'Packaging Dataset' ('number` and 'object' types as variables are supported, but no other e.g. like 'date').
        NOTE: In addition, the common/original intention of using a L.LightningDataModule is to performe data operations on tensors to improve compute performance. For our (and most) tabular datasets we neglect this aspect,
            since we perform data transformations, which are not tensor based. Therefore data preprocessing and transformations are organized within the class methods 'prepare_data' and 'setup',
            based on if they should be performed a single time only or multiple times (e.g. on each split seperately).
        NOTE: Be aware of the status of your pre-processing pipeline / transformers (data they are fit on) - performance optimization vs. final evaluation vs. inference only.
            The stage parameter ('fit' or 'inference') in def _preprocessing_pipeline controls this internal logic.
        NOTE: Training, validation, testing and prediction are triggered by the Lightning Trainer() methods (.fit(), .validate(), .test() and .predict()).
            The stage parameter ('fit', 'validate', 'test' and 'predict') controles the internal logic to provide the correct data splitting and dataloader generation.

        Args:
            data_dir (str): The directory where the data is stored.
            continuous_cols (List[str], optional): A list of column names for continuous variables. Defaults to None.
            categorical_cols (List[str], optional): A list of column names for categorical variables. Defaults to None.
            target (List[str], optional): A list of column names for the target variable. Defaults to None.
            oversampling (bool, optional): Whether to perform oversampling. Defaults to False.
            task_dataset (str, optional): The type of task dataset. Defaults to 'classification'.
            test_size (Optional[float], optional): The size of the test set. Defaults to None.
            val_size (Optional[float], optional): The size of the validation set. Defaults to None.
            batch_size (int, optional): The batch size for training. Defaults to 64.
            batch_size_inference (Optional[int], optional): The batch size for inference. Defaults to None.
            num_workers_train (int, optional): The number of workers for training. Defaults to 0, which is the main thread (always recoommended).
            num_workers_predict (int, optional): The number of workers for inference. Defaults to 0, which is the main thread (always recoommended).
            kwargs_dataloader_trainvaltest (Dict, optional): Additional keyword arguments for the dataloader for training, validation, and testing. Defaults to {}.
            kwargs_dataloader_predict (Dict, optional): Additional keyword arguments for the dataloader for prediction. Defaults to {}.
            SEED (Optional[int], optional): The seed for reproducibility. Defaults to 42.
        """
        super().__init__()
        # self.save_hyperparameters()
        self.data_dir = data_dir
        self.categorical_cols = categorical_cols if categorical_cols else []
        self.continuous_cols = continuous_cols if continuous_cols else []
        self.task_dataset = task_dataset
        self.task = task_dataset
        self.test_size = test_size
        self.val_size = val_size
        self.target = target
        self.oversampling = oversampling
        self.batch_size = batch_size
        self.batch_size_inference = self.batch_size if not batch_size_inference else batch_size_inference
        self.num_workers_train = num_workers_train
        self.num_workers_predict = num_workers_predict
        self.kwargs_dataloader_trainvaltest = kwargs_dataloader_trainvaltest
        self.kwargs_dataloader_predict = kwargs_dataloader_predict
        self.stage_setup = None
        self.SEED = SEED

        self._prepare_data_called = False

    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Performs general, use case independent data input handeling and data type handling.
        Used internal in 'prepare_data' for train, val and test dataloaders and in 'inference_dataloader' for prediction.
        Target specific handelings are performed in 'perpare_data' to avoid conflicts during inference only scenarios where the target is not available.
        General data preparation involves:
            - transform target variable to data type 'object' for classificatiomn tasks and to data type 'float32' for regression tasks.
            - transform continuous feature variables to data type 'np.float32'.
            - transform categorical feature variables to data type 'object'.
            - update the processed dataframe accordingly and drops not specified columns.
        """
        if self.task == "classification":
            # transform target variable to data type 'object'
            data[self.target] = data[self.target].astype("object").values
        elif self.task == "regression":
            # transform target variable to data type 'float32'
            data[self.target] = data[self.target].astype(np.float32).values

        if len(self.continuous_cols) > 0:
            # continuous_cols will be transfomred to float32 ('32' for performance reasons) since NNs do not handle int properly.
            data[self.continuous_cols] = data[self.continuous_cols].astype(np.float32).values
        if len(self.categorical_cols) > 0:
            # ensure that all categorical variables are of type 'object'
            data[self.categorical_cols] = data[self.categorical_cols].astype('object').values

        if (len(self.continuous_cols) > 0) or (len(self.categorical_cols) > 0):
            self.feature_cols = self.continuous_cols + self.categorical_cols
            pass
        else:
            raise TypeError("Missing required argument: 'continuous_cols' and/or 'categorical_cols'")

        # Define a subset based on continuous_cols and categorical_cols
        data = data[self.continuous_cols + self.categorical_cols + self.target]

        return data

    def _preprocessing_pipeline(
        self, X: pd.DataFrame = None, y: pd.DataFrame = None, stage: str = "fit"
    ) -> pd.DataFrame:
        """
        PREPROCESSING PIPELINE, used internal in 'setup' for train, val and test dataloaders and in 'inference_dataloader',
        as well as for inverse transformations.
        TabularDatasetPACKAGING prepares data for prediction only accordingly to support _preprocessing_pipeline.
        """
        # create pipeline for fit scenario, use existing pipeline for inference scenario
        if stage == "fit":
            # numerical feature processing
            numerical_features = X.select_dtypes(include='number').columns.tolist()
            numeric_feature_pipeline = Pipeline(steps=[
                ('impute', SimpleImputer(strategy='median')),
                ('scale', StandardScaler())
            ])
            # categorical feature processing
            categorical_features = X.select_dtypes(exclude='number').columns.tolist()
            categorical_feature_pipeline = Pipeline(steps=[
                ('impute', SimpleImputer(strategy='most_frequent')),
                ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)), # ordinal is used instead of label encoder to avoid conflicts with inference or
                ('nan_label', OrdinalEncoderExtensionUnknowns()),
            ])
            # apply both pipeline on seperate columns using "ColumnTransformer"
            self.preprocess_pipeline = ColumnTransformer(transformers=[
                ('number', numeric_feature_pipeline, numerical_features),
                ('category', categorical_feature_pipeline, categorical_features)],
                verbose_feature_names_out=False)
            self.preprocess_pipeline.set_output(transform="pandas")

            # ordinal is used instead of label encoder to avoid conflicts with inference or
            # conflicts caused by data splits of categories with low numerber of classesonly scenarios
            self.label_encoder_target = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            # self.label_encoder_target = LabelEncoder()

        if stage == "fit":
            X_transformed = self.preprocess_pipeline.fit_transform(X)
            y_transformed = pd.DataFrame(data=self.label_encoder_target.fit_transform(y.values.reshape(-1, 1)), index=y.index, columns=y.columns)
        elif stage == 'inference':
            X_transformed = self.preprocess_pipeline.transform(X)
            y_transformed = pd.DataFrame(data=self.label_encoder_target.transform(y.values.reshape(-1, 1)), index=y.index, columns=y.columns)
        else:
            raise ValueError(f"Missing required argument 'stage', must be 'fit' or 'inference', got {stage}")

        return pd.concat([X_transformed, y_transformed], axis=1)

    def prepare_data(self, shuffle: bool = False) -> None:
        """Custom data specific operations and basic tabular specific operations that only should be performed once on the data (and should not be performed on a distributed manner).
        Load the data as Tabular Data as a Pandas DataFrame from a .csv file and performs custom data processing related to loading a .csv file (data type correction) and defining a subset of features.
        In addition "_prepare_data" performace general data preparation for the classification/regression task and perform basic data error handeling. General data preparation involves:
            - transform target variable to data type 'object'.
            - update the processed dataframe accordingly and drops not specified columns.
            - shuffle the data (rows).
        """

        # USE CASE SPECIFIC DATA HANDLING
        self.data = pd.read_csv(self.data_dir, sep="\t")
        # for inference mode, as the target might not be provided in the data, ensures pre-processing pipeline completes correctly.
        if 'packaging_category' not in self.data.columns:
            self.data.insert(len(self.data.columns), 'packaging_category', np.nan) # Insert an empty column at the end (position=-1)
        # define the subset
        self.data = self.data[[
            'material_number',
            'brand',
            'product_area',
            'core_segment',
            'component',
            'manufactoring_location',
            'characteristic_value',
            'material_weight',
            'packaging_code',
            'packaging_category',
        ]]
        self.data['material_number'] = self.data['material_number'].astype('object')

        if self.oversampling:
            # NOTE: Oversampling so each class has at least 100 sample; to properly represent minority classes during training and evaluation
            X = self.data.iloc[:, :-1]
            y = self.data.iloc[:, -1]  # the last column is the target, ensured based on section before (# select a subset)
            dict_oversmapling = {
                "Metal Cassette": 100,
                "Carton tube with or w/o": 100,
                "Wooden box": 100,
                "Fabric packaging": 100,
                "Book packaging": 100,
            }
            oversampler = RandomOverSampler(sampling_strategy=dict_oversmapling, random_state=self.SEED)
            X_oversample, y_oversample = oversampler.fit_resample(X, y)
            self.data = pd.concat([X_oversample, y_oversample], axis=1)

        # GENERAL DATA HANDLING
        self.data = self._prepare_data(self.data)

        # shuffle data
        if shuffle is True: self.data = self.data.sample(frac=1)

        self.n_samples = self.data.shape[0]

        self._prepare_data_called = True

    def setup(self, stage: str = None) -> None:
        """Data Operations (like shuffle, split data, categorical encoding, normalization, etc.) that will be performed multiple times, which any dataframe should undergo before feeding into the dataloader.
        Since on tabular data, operations like transformations (categorical encoding, normalization, etc.) needs to be performed with respect to all samples (respectively separat per train, val, test split),
        most operations are not performed in DDP way. See class docstring for further details regarding tabular data and tensor transformations.

        Args:
            test_size (Optional[float], optional):
                Defines the hold out split that should be used for final performance evaluation. If 'None' no split will be performed and all data is used in 'fit'
            val_size (Optional[float], optional):
                Defines an additional split on the train data that should be used for model optimization. If 'None' no val split will be performed and all train data is used in 'fit'
            stage (Optional[str], optional):
                Internal parameter to distinguish between 'fit', 'validate', 'test' and 'predict'. Defaults to None.
        """

        self.stage_setup = stage

        if not self._prepare_data_called:
            raise RuntimeError("'prepare_data' needs to be called before 'setup'")

        # Define features and target
        X = self.data.iloc[:, :-1]
        y = self.data.iloc[:, -1]  # the last column is the target, ensured by calling 'prepare_data' upfront

        # Define data for train, val and test and for prediction
        if stage in ("fit", "validate", "test"):
            # Generate train, val and test data splits
            if self.test_size is not None:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=self.test_size, stratify=y, random_state=self.SEED
                )
                X_train = pd.DataFrame(data=X_train, columns=X.columns)
                y_train = pd.DataFrame(data=y_train, columns=[y.name])
                X_test = pd.DataFrame(data=X_test, columns=X.columns)
                y_test = pd.DataFrame(data=y_test, columns=[y.name])
                if (self.val_size is not None) and (self.test_size is not None):
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_train, y_train, test_size=self.val_size, stratify=y_train, random_state=self.SEED
                    )
                    X_train = pd.DataFrame(data=X_train, columns=X.columns)
                    y_train = pd.DataFrame(data=y_train, columns=[y.name])
                    X_val = pd.DataFrame(data=X_val, columns=X.columns)
                    y_val = pd.DataFrame(data=y_val, columns=[y.name])
            else:
                X_train = X
                y_train = pd.DataFrame(data=y, columns=[y.name])
        elif stage == "predict":
            X_pred = X
            y_pred = pd.DataFrame(data=y, columns=[y.name])
        else:
            raise ValueError(f"Stage must be 'fit', 'validate', 'test' or 'predict', got {stage}")

        # pre-process data
        if stage in ("fit", "validate", "test"):
            # the logic ensures that y_train is during all training scenarios and inference scenario always the right reference for number of classes.
            tabular_train = self._preprocessing_pipeline(X_train, y_train, stage="fit")
            if self.test_size is not None:
                tabular_test = self._preprocessing_pipeline(X_test, y_test, stage='inference')
            if (self.val_size is not None) and (self.test_size is not None):
                tabular_val = self._preprocessing_pipeline(X_val, y_val, stage='inference')
            # n_classes is calculated based on set of unique classes in train, val, test after preprocessing to handle unknown classes properly.
            self.n_classes = len(
                set(tabular_train[y.name].unique())
                .union(*(set(df[y.name].unique()) for df in (tabular_val, tabular_test) if df is not None))
            )
        elif stage == 'predict':
            tabular_predict = self._preprocessing_pipeline(X_pred, y_pred, stage='inference')

        # create datasets
        # NOTE: instanziation of datasets (train, val test) in stage == ('fit', 'validate', 'test') is controlled by self.test_size and self.val_size
        #       instanziation of datasets (predict) is controlled by stage == 'predict'
        if stage in ("fit", "validate", "test"):
            self.train_dataset = TabularDataset(
                data=tabular_train,
                continuous_cols=self.continuous_cols,
                categorical_cols=self.categorical_cols,
                target=self.target,
                task=self.task_dataset,
            )
            if self.test_size is not None:
                self.test_dataset = TabularDataset(
                    data=tabular_test,
                    continuous_cols=self.continuous_cols,
                    categorical_cols=self.categorical_cols,
                    target=self.target,
                    task=self.task_dataset,
                )
            if (self.val_size is not None) and (self.test_size is not None):
                self.val_dataset = TabularDataset(
                    data=tabular_val,
                    continuous_cols=self.continuous_cols,
                    categorical_cols=self.categorical_cols,
                    target=self.target,
                    task=self.task_dataset,
                )
        elif stage == "predict":
            self.predict_dataset = TabularDataset(
                data=tabular_predict,
                continuous_cols=self.continuous_cols,
                categorical_cols=self.categorical_cols,
                target=self.target,
                task=self.task_dataset,
            )
        else:
            raise ValueError(f"Stage must be 'fit', 'validate', 'test' or 'predict', got {stage}")

    def train_dataloader(self) -> DataLoader:
        """Dataloader that the Trainer fit() method uses.
        Args:
            batch_size (Optional[int], optional): Batch size. Defaults to `self.batch_size`.
        Returns:
            DataLoader: Train dataloader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers_train,
            **self.kwargs_dataloader_trainvaltest,
        )

    def val_dataloader(self) -> DataLoader:
        """Dataloader that the Trainer fit() and validate() methods uses.
        Args:
            batch_size (Optional[int], optional): Batch size. Defaults to `self.batch_size`.
        Returns:
            DataLoader: Validation dataloader
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers_train,
            **self.kwargs_dataloader_trainvaltest,
        )

    def test_dataloader(self) -> DataLoader:
        """Dataloader that the Trainer test() method uses.
        Args:
            batch_size (Optional[int], optional): Batch size. Defaults to `self.batch_size`.
        Returns:
            DataLoader: Test dataloader
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers_train,
            **self.kwargs_dataloader_trainvaltest,
        )

    def predict_dataloader(self) -> DataLoader:
        """Dataloader that the Trainer predict() method uses.
        Used for predictions for data with unknow target (labes).
        Args:
            batch_size (Optional[int], optional): Batch size. Defaults to `self.batch_size`.
        Returns:
            DataLoader: Test dataloader
        """
        if self.stage_setup == 'predict':
            return DataLoader(
                self.predict_dataset,
                batch_size=self.batch_size_inference,
                shuffle=False,
                num_workers=self.num_workers_predict,
                **self.kwargs_dataloader_predict,
            )
        else:
            return None


class MulticlassTabularLightningModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module = None,
        learning_rate: float = 0.001,
        train_acc: Metric = None,
        val_acc: Metric = None,
        test_acc: Metric = None,
    ) -> None:
        """LightningModule for multiclass classification.
        Args:
            n_classes (int): Number of classes.
            model (nn.Module): Model to be trained.
            learning_rate (float): Learning rate.
            train_acc (Metric): Metric for training loss/accuracy.
            val_acc (Metric): Metric for validation loss/accuracy.
            test_acc (Metric): Metric for test loss/accuracy.
        """
        super().__init__()
        # self.save_hyperparameters()
        self.model = model
        self.learning_rate = learning_rate
        self.train_acc = train_acc
        self.val_acc = val_acc
        self.test_acc = test_acc

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through the MLP."""
        return self.model(x)

    def _shared_step(self, batch: Dict[str, torch.Tensor], batch_idx) -> Tuple[torch.Tensor]:
        x = {key: batch[key] for key in ["continuous", "categorical"]}
        y = batch["target"].flatten()  # flatten to match input shape of F.cross_entropy
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        y_hat = torch.argmax(y_hat, dim=1) # provides the class with the highest probability to match the shape of y
        return (loss, y_hat, y)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        loss, y_hat, y = self._shared_step(batch, batch_idx)
        self.log(f"train_loss", loss)
        self.train_acc(y_hat, y)
        self.log("train_F1_macro_weighted", self.train_acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        loss, y_hat, y = self._shared_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        self.val_acc(y_hat, y)
        self.log("val_F1_macro_weighted", self.val_acc, prog_bar=True)
        return

    def test_step(self, batch, batch_idx) -> None:
        _, y_hat, y = self._shared_step(batch, batch_idx)
        self.test_acc(y_hat, y)
        self.log("test_F1_macro_weighted", self.test_acc)
        return

    def predict_step(self, batch, batch_idx) -> torch.Tensor:
        x = {key: batch[key] for key in ["continuous", "categorical"]}
        y_hat = self.forward(x)
        preds = torch.argmax(y_hat, dim=1)
        return preds

    def configure_optimizers(self) -> torch.optim.Adam:
        return torch.optim.Adam(params=self.parameters(), lr=self.learning_rate)
