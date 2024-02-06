
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import numpy as np
from functools import partial


def collate_pair(x: tf.Tensor, pred_len: int, values_idxs: list, label_idxs: list, selection_idxs: tf.Tensor = None, keep_dims: bool = False) -> tuple:
    """
    Collate input data into pairs of selected inputs and corresponding outputs.

    Args:
        x (tf.Tensor): Input data.
        pred_len (int): Prediction length.
        values_idxs (list): Indices of value columns.
        label_idxs (list): Indices of label columns.
        selection_idxs (tf.Tensor, optional): Indices of selected features. Defaults to None.
        keep_dims (bool, optional): Whether to keep dimensions when selecting features. Defaults to False.

    Returns:
        tuple: Selected inputs and outputs as tensors.
    """
    seq_len = len(x)-pred_len
    inputs = x[:-pred_len]

    feat_size = len(label_idxs) + len(values_idxs)

    selected_inputs = tf.gather(tf.reshape(
        inputs, [seq_len*feat_size]), selection_idxs)
    

    if keep_dims:
        padded_selection = tf.zeros_like(
            tf.reshape(inputs, [seq_len*feat_size]))
        padded_selection = tf.tensor_scatter_nd_add(
            padded_selection, selection_idxs.reshape(-1, 1), selected_inputs)
        selected_inputs = tf.reshape(padded_selection, [seq_len, feat_size])

    outputs = tf.squeeze(tf.reshape(
        tf.gather(x[-pred_len:], [label_idxs], axis=1), [pred_len*len(label_idxs)]))
    return selected_inputs, outputs


def batch(seq_len: int, x: tf.Tensor) -> tf.Tensor:
    """
    Batch the input data with a specified sequence length.

    Args:
        seq_len (int): Sequence length.
        x (tf.Tensor): Input data.

    Returns:
        tf.Tensor: Batches of data.
    """
    return x.batch(seq_len)


def get_values_and_labels_index(data: np.ndarray) -> tuple:
    """
    Get the indices of value and label columns in the input data.

    Args:
        data (np.ndarray): Input data.

    Returns:
        tuple: Indices of label columns and value columns.
    """
    label_idxs = [idx for idx, col in enumerate(
        data.drop('year', axis=1).columns) if 'target' in col]
    values_idxs = [idx for idx, col in enumerate(
        data.drop('year', axis=1).columns) if 'target' not in col]

    return label_idxs, values_idxs


def scale(train_df: np.ndarray, valid_df: np.ndarray, test_df: np.ndarray) -> tuple:
    """
    Scale the input datasets using StandardScaler.

    Args:
        train_df (np.ndarray): Training dataset.
        valid_df (np.ndarray): Validation dataset.
        test_df (np.ndarray): Test dataset.

    Returns:
        tuple: Scaled training, validation, and test datasets, and the scaler object.
    """

    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(train_df)

    valid_scaled = scaler.transform(valid_df)

    test_scaled = scaler.transform(test_df)

    return train_scaled, valid_scaled, test_scaled, scaler


def split(data: np.ndarray, parameters: dict) -> tuple:
    """
    Split the data into training, validation, and test datasets based on the given parameters.

    Args:
        data (np.ndarray): Input data.
        parameters (dict): Model parameters.

    Returns:
        tuple: Training, validation, and test datasets.
    """

    input_columns = [col for col in data.columns.tolist() if col != 'year']

    #split_by_year = parameters['dataset']['params'].get('crossval', False)
    test_year = parameters['dataset']['params'].get('test_year', None)

    if test_year != None:
        first_year = min(data.year.unique())

        val_year = test_year-1 if test_year > first_year else first_year+1

        train_df = data.loc[~data.year.isin(
            [test_year, val_year]), input_columns].values

        valid_df = data.loc[data.year == val_year, input_columns].values

        test_df = data.loc[data.year == test_year, input_columns].values
    else:

        data = data.loc[:, input_columns]
        train_df = data.iloc[:int(len(data)*0.8)].values

        valid_df = data.iloc[int(len(data)*0.8):int(len(data)*0.9)].values

        test_df = data.iloc[int(len(data)*0.9):].values

    return train_df, valid_df, test_df


def windowing(train_scaled: np.ndarray, valid_scaled: np.ndarray, test_scaled: np.ndarray, values_idxs: list, label_idxs: list, selection_idxs: tf.Tensor, parameters: dict) -> tuple:
    """
    Prepare the data for windowing and batching.

    Args:
        train_scaled (np.ndarray): Scaled training dataset.
        valid_scaled (np.ndarray): Scaled validation dataset.
        test_scaled (np.ndarray): Scaled test dataset.
        values_idxs (list): Indices of value columns.
        label_idxs (list): Indices of label columns.
        selection_idxs (tf.Tensor): Indices of selected features.
        parameters (dict): Model parameters.

    Returns:
        tuple: Training, validation, and test datasets in the specified format.
    """
    seq_len = parameters['dataset']['params']['seq_len']
    pred_len = parameters['dataset']['params']['pred_len']
    shift = parameters['dataset']['params']['shift'] or seq_len
    select_timesteps = parameters['dataset']['params'].get('select_timesteps', True)
    model_type = parameters['model']['params']['type']
    batch_size = parameters['model']['params']['batch_size']

    keep_dims = parameters['model']['params'].get('keep_dims', False)

    data_train = tf.data.Dataset.from_tensor_slices(train_scaled)
    data_valid = tf.data.Dataset.from_tensor_slices(valid_scaled)
    data_test = tf.data.Dataset.from_tensor_slices(test_scaled)

    batch_seq = partial(batch, seq_len+pred_len)
    data_train = data_train.window(seq_len+pred_len, shift=shift, drop_remainder=True).flat_map(batch_seq).map(
        lambda x: collate_pair(x, pred_len, values_idxs, label_idxs, selection_idxs, keep_dims))
    data_valid = data_valid.window(seq_len+pred_len, shift=shift, drop_remainder=True).flat_map(batch_seq).map(
        lambda x: collate_pair(x, pred_len, values_idxs, label_idxs, selection_idxs, keep_dims))
    data_test = data_test.window(seq_len+pred_len, shift=shift, drop_remainder=True).flat_map(batch_seq).map(
        lambda x: collate_pair(x, pred_len, values_idxs, label_idxs, selection_idxs, keep_dims))

    if model_type == 'tensorflow':
        data_train = data_train.batch(
            batch_size, drop_remainder=True).cache().prefetch(tf.data.AUTOTUNE)
    else:
        data_train = list(map(lambda x: x.numpy(), next(
            data_train.batch(999999).__iter__())))

    data_valid = list(map(lambda x: x.numpy(), next(
        data_valid.batch(999999).__iter__())))
    data_test = list(map(lambda x: x.numpy(), next(
        data_test.batch(999999).__iter__())))
    
    return data_train, data_valid, data_test


def get_feature_names(data, parameters):

    seq_len = parameters['dataset']['params']['seq_len']
    select_timesteps = parameters['dataset']['params'].get('select_timesteps', True)

    feature_names = np.array(
        [col for col in data.drop('year', axis=1).columns])

    features = np.array([np.core.defchararray.add(
        feature_names, ' t-'+str(i)) for i in range(seq_len, 0, -1)]).flatten()

    return features
