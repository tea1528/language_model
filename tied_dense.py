import keras
from keras import backend as K
from keras.layers import Layer, Dense
import keras.activations as activations
from keras.engine.base_layer import InputSpec



class TiedEmbeddingsTransposed(Layer):
    """Layer for tying embeddings in an output layer.
    A regular embedding layer has the shape: V x H (V: size of the vocabulary. H: size of the projected space).
    In this layer, we'll go: H x V.
    With the same weights than the regular embedding.
    In addition, it may have an activation.
    # References
        - [ Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859)
    """

    def __init__(self, tied_to=None,
                 activation=None,
                 **kwargs):
        super(TiedEmbeddingsTransposed, self).__init__(**kwargs)
        self.tied_to = tied_to
        self.activation = activations.get(activation)

    def build(self, input_shape):
        self.kernel = K.transpose(self.tied_to.get_weights()[0])
        self.built = True

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape[0], K.int_shape(K.constant(self.tied_to.get_weights()[0]))[0]

    def call(self, inputs, mask=None):
        output = K.dot(inputs, self.kernel)
        if self.activation is not None:
            output = self.activation(output)
        return output


    def get_config(self):
        config = {'activation': activations.serialize(self.activation)                  
                 }
        base_config = super(TiedEmbeddingsTransposed, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


    
class DenseTranspose(Dense):
    """
    A Keras dense layer that has its weights set to be the transpose of 
    another layer. Used for implemeneting BidNNs.
    """
    def __init__(self, other_layer=None, activation=None, **kwargs):
        super().__init__(other_layer.input_dim, **kwargs)
        self.other_layer = other_layer
        self.activation = activations.get(activation)
        

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.input_dim = input_dim
        self.input_spec = [InputSpec(dtype=K.floatx(), ndim='2+')]

        self.kernel = K.transpose(self.other_layer.get_weights()[0]) 

        if self.use_bias:
            self.bias = self.add_weight((self.units,),
                                     initializer=self.bias_initializer,
                                     name='bias',
                                     regularizer=self.bias_regularizer,
                                     constraint=self.bias_constraint)
        else:
            self.bias = None

        self.built = True
        
    def get_config(self):
        config = {'activation': activations.serialize(self.activation),
                  'other_layer': self.other_layer
                 }
        base_config = super(DenseTranspose, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
#     def call(self, inputs, mask=None):
#         output = K.dot(inputs, self.kernel)
#         if self.activation is not None:
#             output = self.activation(output)
#         return output
    
#     def get_config(self):
#         config = {'activation': activations.serialize(self.activation),
#                   'other_layer': self.other_layer
#                  }
#         base_config = super(DenseTranspose, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))

