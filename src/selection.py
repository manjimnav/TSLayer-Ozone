from sklearn.feature_selection import VarianceThreshold, SelectPercentile, f_regression, mutual_info_regression, f_classif, mutual_info_classif
import numpy as np

def window_stack(a, stepsize=1, width=3):
    return np.hstack( [a[i:1+i-width or None:stepsize] for i in range(0,width)] )

def get_features_scores_sum(scores, total_sum=0.9):
    scores = scores/scores.sum(axis=0,keepdims=1)
    sort_index = np.argsort(-scores)
    cumsum = np.cumsum(scores[sort_index])
    limit = np.argmax(cumsum>total_sum)
    return sort_index[:limit+1]

def correlation_selection(features, labels, total_sum=0.9):
    correlations = []
    for f_idx in range(features.shape[1]):
        correlation = np.corrcoef(features[:, f_idx], labels.squeeze())[0, 1]
        correlations.append(correlation)
    
    return get_features_scores_sum(np.abs(np.array(correlations)), total_sum=total_sum)

def get_selected_indexes(selection_method, features, labels, original_indexes, parameters):

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

def select_features(data, parameters, labels_idxs):
    seq_len = parameters['dataset']['params']['seq_len']
    pred_len = parameters['dataset']['params']['pred_len']
    shift = parameters['dataset']['params']['shift']
    select_timesteps = parameters['dataset']['params']['select_timesteps']
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

    if select_timesteps:
        return selected_indexes
    else: # Select the features which select at least more than half timesteps

        selection_mask = np.zeros(seq_len*data.shape[1])
        selection_mask[selected_indexes] = 1

        features_selected_mask = selection_mask.reshape(seq_len, -1).sum(axis=0)>(seq_len//2)
        
        selected_indexes = np.arange(data.shape[1])[features_selected_mask]
        return selected_indexes        