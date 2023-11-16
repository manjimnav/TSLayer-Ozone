import yaml
from itertools import product
import numpy as np
import pandas as pd
from .dataset import get_values_and_labels_index, split, scale, windowing, get_feature_names
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from .selection import select_features
from .model import get_model, get_selected_idxs
from pathlib import Path
import tensorflow as tf
import time
from IPython.display import display, clear_output
import random
import os
import hashlib
import json
from copy import deepcopy
from bayes_opt import BayesianOptimization, UtilityFunction
from datetime import datetime, timedelta
from typing import Tuple, Iterable, Union, Optional
from sklearn.base import BaseEstimator


class ExperimentInstance:

    """
    Initialize an ExperimentInstance which runs a single experiment with a set of parameters.

    Args:
        parameters (dict): Experiment parameters.
    """

    def __init__(self, parameters) -> None:
        
        self.parameters = parameters
        self.metrics = pd.DataFrame()
        self.data = None
        self.scaler = None
        self.model = None
        self.dataset = self.parameters['dataset']['name']
        self.label_idxs, self.values_idxs = [], []
        self.code = self.dict_hash(parameters)
        self.selected_idxs = []
        self.raw_results_ = []
        self.data_train = None

        parent_directory = Path(__file__).resolve().parents[1]
        self.dataset_path = f'{parent_directory}/data/processed/{self.dataset}/data.csv'

    def convert(self, num):
        if isinstance(num, np.int64) or isinstance(num, np.int32): return int(num)  
        raise TypeError

    def dict_hash(self, dictionary:dict) -> str:
        """
        MD5 hash of the parameters used as experiment identifier.
        
        """
        dhash = hashlib.md5()
        # We need to sort arguments so {'a': 1, 'b': 2} is
        # the same as {'b': 2, 'a': 1}
        print(dictionary)
        encoded = json.dumps(dictionary, sort_keys=True, default=self.convert).encode()
        dhash.update(encoded)
        return dhash.hexdigest()

    def preprocess_data(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Preprocess the data for training.

        Returns:
            Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: Training, validation, and test datasets.
        """
        self.label_idxs, self.values_idxs = get_values_and_labels_index(self.data)
        if len(self.values_idxs)<1 and not self.parameters['dataset']['params'].get('select_timesteps', True):
            raise Exception(f"Cannot select features in dataset {self.parameters['dataset']['name']}")
            
        train_df, valid_df, test_df = split(self.data, self.parameters)

        train_scaled, valid_scaled, test_scaled, self.scaler = scale(train_df, valid_df, test_df)

        self.selected_idxs = select_features(train_scaled, self.parameters, self.label_idxs)

        data_train, data_valid, data_test = windowing(train_scaled, valid_scaled, test_scaled, self.values_idxs, self.label_idxs, self.selected_idxs, self.parameters)
        
        self.data_train = np.concatenate(list(map(lambda x: x.numpy(), next(data_train.batch(9999999999).__iter__())))[0])
        
        return data_train, data_valid, data_test

    def read_data(self) -> None:
        """Read data from the dataset path."""
        self.data = pd.read_csv(self.dataset_path)
    
    def recursive_items(self, dictionary, parent_key=None) -> Iterable:
        """
        Recursively iterate over dictionary items.

        Args:
            dictionary (dict): The input dictionary.
            parent_key (str, optional): The parent key in the recursion. Defaults to None.

        Yields:
            Iterable: Key-value pairs.
        """
        for key, value in dictionary.items():
            key = key if parent_key is None else parent_key+'_'+key
            if type(value) is dict:
                yield from self.recursive_items(value, parent_key=key)
            else:
                yield (key, value)

    
    def train_tf(self, model: tf.keras.Model, data_train: tf.data.Dataset, data_valid: tf.data.Dataset) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
        """
        Train a TensorFlow model.

        Args:
            model (tf.keras.Model): The model to train.
            data_train (tf.data.Dataset): Training data.
            data_valid (tf.data.Dataset): Validation data.

        Returns:
            Tuple[tf.keras.Model, tf.keras.callbacks.History]: Trained model and training history.
        """

        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

        history = model.fit(
            data_train,
            epochs=100,
            callbacks=[callback],
            validation_data=data_valid,
            verbose = 0
        )

        if 'TimeSelectionLayer' in self.parameters['selection']['name']:
            self.selected_idxs = get_selected_idxs(model, get_feature_names(self.data, self.parameters))

        return model, history
    
    def train_sk(self, model: BaseEstimator, data_train: np.ndarray) -> Tuple[BaseEstimator, None]:
        """
        Train a scikit-learn model.

        Args:
            model: The scikit-learn model to train.
            data_train (tuple): Training data as a tuple of inputs and outputs.

        Returns:
            Tuple[Any, None]: Trained model and None (no training history).
        """

        model_name = self.parameters['model']['name']

        model.fit(data_train[0], data_train[1])

        features = get_feature_names(self.data, self.parameters)
        features_idxs = np.arange(0, features.flatten().shape[0])

        if model_name == 'lasso':
            importances = model.coef_.max(axis=0)
        else:
            importances = model.feature_importances_
        self.selected_idxs = features_idxs[importances>0]

        return model, None

    def train(self, model: Union[tf.keras.Model, BaseEstimator], data_train, data_valid) -> Tuple[Union[tf.keras.Model, BaseEstimator], Optional[tf.keras.callbacks.History]]:
        """
        Train a model.

        Args:
            model: The model to train.
            data_train: Training data.
            data_valid: Validation data.

        Returns:
            Tuple[Union[tf.keras.Model, Any], Optional[tf.keras.callbacks.History]]: Trained model and training history (if available).
        """

        model_type = self.parameters['model']['params']['type']

        if model_type == 'tensorflow':
            model, history = self.train_tf(model, data_train, data_valid)
        else:
            model, history = self.train_sk(model, data_train)

        return model, history
        

    def calculate_metrics(self, model: Union[tf.keras.Model, BaseEstimator], history: tf.keras.callbacks.History, data_test: np.ndarray, data_valid: np.ndarray, duration: float) -> Tuple[pd.DataFrame, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Calculate evaluation metrics.

        Args:
            model: The trained model.
            history: The training history (optional).
            data_test (tuple): Test data as a tuple of inputs and outputs.
            data_valid (tf.data.Dataset): Validation data.
            duration (float): Duration of training.

        Returns:
            Tuple[pd.DataFrame, tf.Tensor, tf.Tensor, tf.Tensor]: Metrics DataFrame, test data inputs, true values, and predictions.
        """

        model_type = self.parameters['model']['params']['type']
        
        mean = self.scaler.mean_[self.label_idxs]
        std = self.scaler.scale_[self.label_idxs]

        predictions = model.predict(data_test[0])
        predictions_valid = model.predict(data_valid[0])

        true = data_test[1]*std + mean
        true_valid = data_valid[1]*std + mean
        predictions = predictions*std + mean
        predictions_valid = predictions_valid*std + mean

        metrics = pd.DataFrame({'mean_squared_error': mean_squared_error(true, predictions), 
                                'root_mean_squared_error': np.sqrt(mean_squared_error(true, predictions)),
                'mean_absolute_error': mean_absolute_error(true, predictions),
                'mean_absolute_percentage_error': mean_absolute_percentage_error(true, predictions),
                'r2': r2_score(true, predictions),
                'mean_absolute_error_valid': mean_absolute_error(true_valid, predictions_valid),
                'root_mean_squared_error_valid': np.sqrt(mean_squared_error(true_valid, predictions_valid)),
                'mean_squared_error_valid': mean_squared_error(true_valid, predictions_valid),
                }, index=[0])

        metrics['dataset'] = self.dataset
        for key, value in self.recursive_items(self.parameters):
            metrics[key] = value
        
        original_features = get_feature_names(self.data, self.parameters)
        metrics['features'] = [original_features.tolist()]
        metrics['selected_features'] = [original_features[list(self.selected_idxs)].tolist()]

        metrics['duration'] = duration

        if history is not None:
            metrics['history'] = str(history.history.get('val_loss', None))
            metrics['val_loss'] = min(history.history.get('val_loss', None))

        metrics['code'] = self.code

        true = true.flatten()
        predictions = predictions.flatten()
        
        return metrics, data_test[0], true, predictions
    
    def execute_one(self) -> Tuple[pd.DataFrame, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Execute a single experiment instance.

        Returns:
            Tuple[pd.DataFrame, tf.Tensor, tf.Tensor, tf.Tensor]: Metrics DataFrame, test data inputs, true values, and predictions.
        """
        data_train, data_valid, data_test = self.preprocess_data()

        model = get_model(self.parameters, self.label_idxs, self.values_idxs)

        start = time.time()
        self.model, history = self.train(model, data_train, data_valid)
        duration = time.time() - start

        metrics = self.calculate_metrics(self.model, history, data_test, data_valid, duration)

        return metrics
    
    def run(self) -> pd.DataFrame:
        """
        Run the experiment instance.

        Returns:
            pd.DataFrame: Metrics DataFrame.
        """
        
        self.read_data()

        split_by_year = self.parameters['dataset']['params'].get('crossval', False)

        self.metrics = pd.DataFrame()
        if split_by_year:
            for test_year in sorted(self.data.year.unique()): # yearly crossval
                if test_year<2015: continue
                test_year = self.parameters['dataset']['params']['test_year'] = test_year
                year_metrics, inputs, true, predictions = self.execute_one()

                self.metrics = pd.concat([self.metrics, year_metrics])

                dates = pd.date_range(datetime(test_year, 1, 1) + timedelta(hours=25), datetime(test_year, 12, 31), freq='H')
                dates = dates[:len(true)]
                self.raw_results_.append((dates, inputs, true, predictions))
        else:
            self.metrics, inputs, true, predictions = self.execute_one() 
            #dates = pd.date_range(datetime(test_year, 1, 1) + timedelta(hours=25), datetime(test_year, 12, 31), freq='H')
            #dates = dates[:len(true)]
            self.raw_results_.append((inputs, true, predictions))

        return self.metrics

class ExperimentLauncher:

    """
    Initialize an ExperimentLauncher.

    Args:
        config_path (str): Path to the configuration files.
        save_file (str, optional): Path to save the results CSV file. Defaults to "../results/TimeSelection/results.csv".
        search_type (str, optional): Search type, either 'grid' or 'bayesian'. Defaults to 'grid'.
        iterations (int, optional): Number of iterations for Bayesian optimization. Defaults to 10.
    """

    def __init__(self, config_path, save_file="../results/TimeSelection/results.csv", search_type='grid', iterations=10) -> None:
        
        data_config, selection_config, model_config = f"{config_path}data_config.yaml", f"{config_path}selection_config.yaml", f"{config_path}model_config.yaml"

        self.data_configuration = self.load_config(data_config)
        self.selection_configuration = self.load_config(selection_config)
        self.model_configuration = self.load_config(model_config)
        self.save_file = save_file
        self.search_type = search_type
        self.iterations = iterations
        self.optimizer = None


        if os.path.exists(self.save_file):
            self.metrics = pd.read_csv(self.save_file)
        else:
            self.metrics = pd.DataFrame()
    
    def nested_product(self, configurations: Union[dict, list]) -> Iterable:
        """
        Generate nested product configurations.

        Args:
            configurations: Configuration dictionary.

        Yields:
            Iterable: Nested product configurations.
        """
        if isinstance(configurations, list):
            for value in configurations:
                yield from ([value] if not isinstance(value, (dict, list)) else self.nested_product(value))
        elif isinstance(configurations, dict):
            for key, value in configurations.items():
                if isinstance(value, list) and len(value)==3 and isinstance(value[2], dict) and 'step' in value[2]:
                    configurations[key] = list(np.arange(value[0], value[1]+value[2]['step'], value[2]['step']))
                    
            for i in product(*map(self.nested_product, configurations.values())):
                yield dict(zip(configurations.keys(), i))
        else:
            yield configurations

    def load_config(self, path: str) -> dict:
        """
        Load a YAML configuration file.

        Args:
            path (str): Path to the YAML configuration file.

        Returns:
            dict: Loaded configuration as a dictionary.
        """
        config = {}
        with open(path, "r") as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                raise Exception("Configuration at path {path} was not found")
        return config
    
    def seed(self, seed: int = 123) -> None:
        """
        Seed random number generators for reproducibility.

        Args:
            seed (int, optional): Random seed. Defaults to 123.
        """
        tf.random.set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    def transform_to_bounds(self, general_params: dict) -> dict:
        """
        Transform general parameters to bounds for Bayesian optimization.

        Args:
            general_params (dict): General parameters.

        Returns:
            dict: Bounds for Bayesian optimization.
        """
        bounds = {}
        for general_key in general_params.keys():
            if general_params[general_key]['params'] is not None:
                bound_params = {key: value for key, value in general_params[general_key]['params'].items() if type(value) == list}

                bounds.update(bound_params)
        
        return bounds
    
    def update_params(self, optimized_params: dict, general_params: dict) -> dict:
        """
        Update general parameters with optimized values.

        Args:
            optimized_params (dict): Optimized parameters.
            general_params (dict): General parameters.

        Returns:
            dict: Updated general parameters.
        """
        
        params = deepcopy(general_params)
        for general_key in params.keys():
            if params[general_key]['params'] is not None:
                keys_to_update = list(params[general_key]['params'].keys())
                for key_to_update in keys_to_update:
                    
                    if key_to_update in optimized_params:
                        BuiltinClass = params[general_key]['params'][key_to_update][0].__class__
                        params[general_key]['params'][key_to_update] = BuiltinClass(optimized_params[key_to_update])
        return params
    
    def bayesian_optimization(self, general_params: dict) -> Iterable:
        """
        Perform Bayesian optimization for hyperparameter search.

        Args:
            general_params (dict): General parameters.

        Yields:
            Iterable: Hyperparameter configurations.
        """

        bounds = self.transform_to_bounds(general_params)

        self.optimizer = BayesianOptimization(
            f=None,
            pbounds=bounds,
            verbose=2,
            random_state=1,
        )

        utility = UtilityFunction(kind="ucb", kappa=2.576)

        for _ in range(self.iterations):
            optimized_params = self.optimizer.suggest(utility)      

            params = self.update_params(optimized_params, general_params)

            yield params

    def search_hyperparameters(self, general_params: dict) -> Iterable:
        """
        Search hyperparameters using grid search or Bayesian optimization.

        Args:
            general_params (dict): General parameters.

        Yields:
            Iterable: Hyperparameter configurations.
        """
        if self.search_type == 'grid':
            yield from self.nested_product(general_params)
        elif self.search_type == 'bayesian':
            yield from self.bayesian_optimization(general_params)
        

    def run(self) -> pd.DataFrame:
        """
        Run the experiment launcher.

        Returns:
            pd.DataFrame: Metrics DataFrame.
        """

        for dataset, selection, model in product(self.data_configuration.keys(), self.selection_configuration.keys(), self.model_configuration.keys()):
            
            dataset_params = {"dataset": {'name': dataset, "params": self.data_configuration[dataset]}}
            selection_params = {"selection": {"name": selection, "params": self.selection_configuration[selection]}}
            model_params = {"model": {"name": model, "params": self.model_configuration[model]}}

            general_params = {**dataset_params, **selection_params, **model_params}

            for params in self.search_hyperparameters(general_params):

                if (params['model']['params']['type'] == "sklearn" and params['selection']['name'] != 'NoSelection'):
                    continue   
                
                self.seed()                
                experiment = ExperimentInstance(params)
                
                if experiment.code in self.metrics.get('code', default=pd.Series([], dtype=str)).tolist():
                    print(f"Skipping {params}")
                    if self.optimizer == 'bayesian':
                        self.optimizer.register(params=params, target=-self.metrics.loc[self.metrics.code == experiment.code,'root_mean_squared_error_valid'].mean())
                    continue

                metrics = experiment.run()

                if self.optimizer == 'bayesian':
                    self.optimizer.register(params=params, target=-metrics['root_mean_squared_error_valid'].mean())
            
                self.metrics = pd.concat([self.metrics, metrics])

                self.metrics.to_csv(self.save_file, index=None)
                clear_output(wait=True)
                display(self.metrics)
        
        return self.metrics


