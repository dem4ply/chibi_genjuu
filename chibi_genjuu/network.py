from chibi.object import Chibi_object, descriptor
import yaml
import lasagne
from lasagne import layers
from lasagne import objectives

import theano
import theano.tensor as T

import numpy as np

__all__ = [ 'Network' ]


class Layer( Chibi_object ):
    name = descriptor.String()

    def real_layer( self ):
        raise NotImplementedError

    @property
    def dict( self ):
        return dict( name=self.name )

    @property
    def yaml( self ):
        return yaml.dump( self.dict )


class Input( Layer ):
    batch_size = descriptor.Descriptor()
    shape = descriptor.List()
    input_var = descriptor.Descriptor()

    def __init__(
            self, *args, real_shape=None, shape=None, input_var=None, **kw ):

        if input_var is None:
            input_var = T.lmatrix()
        if isinstance( input_var, str ):
            if input_var == 'int32':
                input_var = T.imatrix()
            else:
                input_var = T.lmatrix()

        super().__init__( shape=shape, input_var=input_var, *args, **kw )
        if not self.shape:
            self.shape = [ None ]

    @property
    def real_layer( self ):
        return layers.InputLayer(
            self.real_shape, input_var=self.input_var, name=self.name )

    @property
    def type_input_var( self ):
        return self.input_var.type.numpy_dtype.name

    @property
    def real_shape( self ):
        return ( self.batch_size, ) + tuple( self.shape )

    @Layer.dict.getter
    def dict( self ):
        result = super().dict
        result.update( dict(
            batch_size=self.batch_size, shape=self.shape,
            input_var=self.type_input_var) )
        return result

    @classmethod
    def from_dict( cls, d ):
        return cls( **d )


class Network( Chibi_object ):
    """
    Contendor de redes neuronales usando el framework lasagne

    Atributes
    ---------
    name: string
    """
    name = descriptor.String( default='test_network' )
