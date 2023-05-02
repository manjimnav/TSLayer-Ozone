import tensorflow as tf
import math

def hard_sigmoid(x):
    return tf.clip_by_value((x + 1.)/2., 0, 1)

def round_through(x):
    '''Element-wise rounding to the closest integer with full gradient propagation.
    A trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182)
    a op that behave as f(x) in forward mode,
    but as g(x) in the backward mode.
    '''
    rounded = tf.round(x)
    return x + tf.stop_gradient(rounded-x)
  
def binary_sigmoid_unit(x):
    return round_through(hard_sigmoid(x))

class TimeSelectionLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs, regularization=0.001, select_timesteps=True, **kwargs):
        super(TimeSelectionLayer, self).__init__( **kwargs)
        self.mask = None
        self.num_outputs = num_outputs
        self.select_timesteps = select_timesteps
        self.regularization = regularization
    
    def custom_regularizer(self, weights):
        weight = self.regularization/math.log((10**self.num_outputs))
        return tf.reduce_sum(weight * binary_sigmoid_unit(weights))

    def build(self, input_shape):
        if self.select_timesteps:
            shape = [int(input_shape[-2]), int(input_shape[-1])]
        else:
            shape = [int(input_shape[-1])]

        self.mask = self.add_weight("kernel",
                                      shape=shape,
                                      initializer=tf.keras.initializers.Constant(value=0.01),
                                      regularizer=self.custom_regularizer)
        
    def get_mask(self):
        
        return binary_sigmoid_unit(tf.expand_dims(self.mask, 0))[0]
        
    def call(self, inputs):

        return tf.multiply(inputs, self.get_mask())