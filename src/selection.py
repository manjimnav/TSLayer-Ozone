from sklearn.feature_selection import VarianceThreshold, SelectPercentile, f_regression, mutual_info_regression
import numpy as np

def window_stack(a: np.ndarray, stepsize: int = 1, width: int = 3) -> np.ndarray:
    """
    Stack overlapping windows of an array.

    Args:
        a (np.ndarray): Input array.
        stepsize (int, optional): Step size between windows. Defaults to 1.
        width (int, optional): Width of each window. Defaults to 3.

    Returns:
        np.ndarray: Stacked windows.
    """
    return np.hstack( [a[i:1+i-width or None:stepsize] for i in range(0,width)] )

def get_features_scores_sum(scores: np.ndarray, total_sum: float = 0.9) -> np.ndarray:
    """
    Get the indices of features that contribute to a specified total sum of scores.

    Args:
        scores (np.ndarray): Feature scores.
        total_sum (float, optional): Target total sum of scores. Defaults to 0.9.

    Returns:
        np.ndarray: Indices of selected features.
    """
    scores = scores/scores.sum(axis=0,keepdims=1)
    sort_index = np.argsort(-scores)
    cumsum = np.cumsum(scores[sort_index])
    limit = np.argmax(cumsum>total_sum)
    return sort_index[:limit+1]

def correlation_selection(features: np.ndarray, labels: np.ndarray, total_sum: float = 0.9) -> np.ndarray:
    """
    Perform feature selection based on feature-label correlations.

    Args:
        features (np.ndarray): Input features.
        labels (np.ndarray): Target labels.
        total_sum (float, optional): Target total sum of correlations. Defaults to 0.9.

    Returns:
        np.ndarray: Indices of selected features.
    """
    correlations = []
    for f_idx in range(features.shape[1]):
        correlation = np.corrcoef(features[:, f_idx], labels.squeeze())[0, 1]
        correlations.append(correlation)
    
    return get_features_scores_sum(np.abs(np.array(correlations)), total_sum=total_sum)


def get_selected_indexes(selection_method: str, features: np.ndarray, labels: np.ndarray,
                         original_indexes: np.ndarray, parameters: dict) -> np.ndarray:
    """
    Get selected feature indices based on the specified selection method.

    Args:
        selection_method (str): The selection method.
        features (np.ndarray): Input features.
        labels (np.ndarray): Target labels.
        original_indexes (np.ndarray): Original feature indices.
        parameters (dict): Model parameters.

    Returns:
        np.ndarray: Selected feature indices.
    """

    selected_indexes = original_indexes
    if 'Variance' in selection_method:
        thr = parameters['selection'].get('params', dict()).get('threshold', None)
        selector = VarianceThreshold()
        selector.fit_transform(features)
        selected_indexes = selector.get_feature_names_out(selected_indexes)
    elif 'Linear' in selection_method:
        thr = parameters['selection'].get('params', dict()).get('threshold', None)
        selector = SelectPercentile(f_regression, percentile=10)
        selector.fit(features, labels)
        scores = np.abs(selector.scores_)
        indexes = get_features_scores_sum(scores, total_sum=thr)
        selected_indexes = indexes
    elif 'MutualInformation' in selection_method:
        thr = parameters['selection'].get('params', dict()).get('threshold', None)
        selector = SelectPercentile(mutual_info_regression, percentile=10)
        selector.fit(features, labels)
        scores = np.abs(selector.scores_)
        indexes = get_features_scores_sum(scores, total_sum=thr)
        selected_indexes = indexes
    elif 'Correlation' in selection_method:
        thr = parameters['selection'].get('params', dict()).get('threshold', None)
        mask = correlation_selection(features, labels, total_sum=thr)
        
        selected_indexes = selected_indexes[mask]
    
    return selected_indexes

def select_features(data: np.ndarray, parameters: dict, labels_idxs: list) -> np.ndarray:
    """
    Select features based on the given data and parameters.

    Args:
        data (np.ndarray): Input data.
        parameters (dict): Model parameters.
        labels_idxs (list): List of label indices.

    Returns:
        np.ndarray: Selected feature indices.
    """
    seq_len = parameters['dataset']['params']['seq_len']
    pred_len = parameters['dataset']['params']['pred_len']
    shift = parameters['dataset']['params']['shift']
    selection_method = parameters['selection']['name']

    original_indexes = np.arange(data.shape[1]*seq_len)
    
    data_windowed = window_stack(data, stepsize=shift, width=seq_len+pred_len)

    features = data_windowed[:, :data.shape[1]*seq_len]
    labels = data_windowed[:, data.shape[1]*seq_len:]

    labels = labels.reshape(-1, pred_len, data.shape[1])[:, :, labels_idxs].reshape(-1, pred_len*len(labels_idxs))

    selected_indexes = set()
    for index in range(labels.shape[1]):
        indexes = get_selected_indexes(selection_method, features, labels[:, index], original_indexes, parameters)

        selected_indexes.update(indexes.tolist())
    
    selected_indexes = np.array(list(selected_indexes))

    return selected_indexes