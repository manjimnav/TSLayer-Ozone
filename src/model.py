from tensorflow import keras
from tensorflow.keras import layers
from functools import partial
from .layer import TimeSelectionLayer, binary_sigmoid_unit, TimeSelectionLayerSmooth, TimeSelectionLayerConstant
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
import numpy as np


def get_hyperparameters():

    activation = 'linear'
    loss = keras.losses.MSE
    metrics = [keras.metrics.MSE, keras.metrics.MAE,
               keras.metrics.mean_absolute_percentage_error]

    return activation, loss, metrics


def get_base_layer(layer_type):
    if layer_type == 'dense':
        layer_base = layers.Dense
    elif layer_type == 'lstm':
        layer_base = partial(layers.LSTM, return_sequences=True)
    elif layer_type == 'cnn':
        layer_base = partial(layers.Conv1D, kernel_size=3)

    return layer_base

def head_layers(parameters, n_features_out, name=''):
    selection = parameters['selection']['name']
    select_timesteps = parameters['dataset']['params']['select_timesteps']
    
    head_layers = []
    if selection == 'TimeSelectionLayer':
        regularization = parameters['selection']['params']['regularization']
        head_layers.append(TimeSelectionLayer(num_outputs=n_features_out,
                           regularization=regularization, select_timesteps=select_timesteps, name=f'{name}'))

    elif selection == 'TimeSelectionLayerSmooth':
        regularization = parameters['selection']['params']['regularization']
        head_layers.append(TimeSelectionLayerSmooth(num_outputs=n_features_out,
                           regularization=regularization, select_timesteps=select_timesteps, name=f'{name}'))
    elif selection == 'TimeSelectionLayerConstant':
        regularization = parameters['selection']['params']['regularization']
        head_layers.append(TimeSelectionLayerConstant(num_outputs=n_features_out,
                           regularization=regularization, select_timesteps=select_timesteps, name=f'{name}'))
    
    if parameters['model']['name'] == 'dense':
        head_layers.append(layers.Flatten())
    
    if len(head_layers)>0:
        return head_layers
    else:
        return None
        
def get_tf_model(parameters, label_idxs, values_idxs):
    model = parameters['model']['name']
    n_layers = parameters['model']['params']['layers']
    n_units = parameters['model']['params']['units']
    dropout = parameters['model']['params']['dropout']
    selection = parameters['selection']['name']
    residual = parameters['selection'].get('params', dict()) or dict()
    residual = residual.get('residual', False)
    pred_len = parameters['dataset']['params']['pred_len']
    seq_len = parameters['dataset']['params']['seq_len']
    select_timesteps = parameters['dataset']['params']['select_timesteps']
    
    activation, loss, metrics = get_hyperparameters()

    n_features_in = len(label_idxs) + len(values_idxs)
    n_features_out = len(label_idxs)
        
    layer_base = get_base_layer(model)
    
    inputs_raw = layers.Input(shape=(seq_len*n_features_in,), name='inputs')
    inputs = layers.Reshape((seq_len, n_features_in), name='inputs_reshaped')(inputs_raw)
    
    header = keras.Sequential(head_layers(parameters, n_features_out*pred_len, name=f'selection_in'))
    
    x = inputs if header is None else header(inputs)
    
    for i in range(n_layers):
        if i > 0 and residual:
            header = keras.Sequential(head_layers(parameters, n_features_out*pred_len, name=f'selection_{i}'))
            formatted_inputs = inputs if header is None else header(inputs)
        
            x = layers.Concatenate()([x, formatted_inputs])
        
        x = layer_base(n_units, activation="relu" if model != 'lstm' else "tanh", name=f"layer{i}")(x)
        x = layers.Dropout(dropout)(x)
    
    if residual:
        header = keras.Sequential(head_layers(parameters, n_features_out*pred_len, name=f'selection_out'))
        formatted_inputs = inputs if header is None else header(inputs)

        x = layers.Concatenate()([x, formatted_inputs])
        
    outputs = layers.Dense(n_features_out*pred_len, name="output")(x)
    model = keras.Model(inputs=inputs_raw, outputs=outputs, name="tsmodel")
        
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=loss,
        metrics=metrics
    )
    
    return model


def get_sk_model(parameters):

    model = parameters['model']['name']

    if model == 'decisiontree':
        model = DecisionTreeRegressor(max_depth=parameters['model']['params']['max_depth'])
    elif model == 'lasso':
        model = Lasso(alpha=parameters['model']['params']['regularization'])
    elif model == 'randomforest':
        model = RandomForestRegressor(max_depth=parameters['model']['params']['max_depth'], n_estimators=parameters['model']['params']['n_estimators'])
    else:
        raise NotImplementedError()

    return model


def get_model(parameters, label_idxs, values_idxs):

    model_type = parameters['model']['params']['type']

    if model_type == 'tensorflow':
        model = get_tf_model(parameters, label_idxs, values_idxs)
    else:
        model = get_sk_model(parameters)

    return model


def get_selected_idxs(model, features):
    
    selected_idxs = set()
    for layer in model.layers:
        if 'selection' in layer.name:
            mask = binary_sigmoid_unit(layer.get_mask()).numpy()
            selected_idxs = selected_idxs.union(np.arange(0, features.flatten().shape[0])[
                mask.flatten().astype(bool)].tolist())
        elif type(layer) == keras.Sequential:
            selected_idxs = selected_idxs.union(get_selected_idxs(layer, features))
    return selected_idxs
