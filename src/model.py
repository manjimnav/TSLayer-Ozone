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


def get_tf_model(parameters, label_idxs, values_idxs):
    model = parameters['model']['name']
    n_layers = parameters['model']['params']['layers']
    n_units = parameters['model']['params']['units']
    dropout = parameters['model']['params']['dropout']
    selection = parameters['selection']['name']
    pred_len = parameters['dataset']['params']['pred_len']
    seq_len = parameters['dataset']['params']['seq_len']
    select_timesteps = parameters['dataset']['params']['select_timesteps']

    activation, loss, metrics = get_hyperparameters()

    n_features_in = len(label_idxs) + len(values_idxs)
    n_features_out = len(label_idxs)

    layer_base = get_base_layer(model)

    layers_list = []

    if model == 'dense':
        layers_list.insert(0, layers.Flatten())

    for i in range(n_layers):
        layers_list.append(layer_base(
            n_units, activation="relu" if model != 'lstm' else "tanh", name=f"layer{i}"))
        layers_list.append(layers.Dropout(dropout))

    layers_list.extend([
        layers.Flatten(),
        layers.Dense(n_features_out*pred_len,
                     activation=activation, name="output")
    ])

    if selection == 'TimeSelectionLayer':
        regularization = parameters['selection']['params']['regularization']
        layers_list.insert(0, TimeSelectionLayer(name='selector', num_outputs=n_features_out,
                           regularization=regularization, select_timesteps=select_timesteps))
        layers_list.insert(0, layers.Reshape((seq_len, n_features_in)))

    elif selection == 'TimeSelectionLayerSmooth':
        regularization = parameters['selection']['params']['regularization']
        layers_list.insert(0, TimeSelectionLayerSmooth(name='selector', num_outputs=n_features_out,
                           regularization=regularization, select_timesteps=select_timesteps))
        layers_list.insert(0, layers.Reshape((seq_len, n_features_in)))
    elif selection == 'TimeSelectionLayerConstant':
        regularization = parameters['selection']['params']['regularization']
        layers_list.insert(0, TimeSelectionLayerConstant(name='selector', num_outputs=n_features_out,
                           regularization=regularization, select_timesteps=select_timesteps))
        layers_list.insert(0, layers.Reshape((seq_len, n_features_in)))

    if model == 'lstm':
        layers_list.insert(0, keras.Input(shape=(seq_len, n_features_in)))

    model = keras.Sequential(layers_list)

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=loss,
        metrics=metrics,
        run_eagerly=True
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
    mask = binary_sigmoid_unit(model.get_layer(
        name="selector").get_mask()).numpy()
    selected_idxs = np.arange(0, features.flatten().shape[0])[
        mask.flatten().astype(bool)].tolist()

    return selected_idxs
