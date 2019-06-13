from chibi.object import Chibi_object, descriptor
import yaml
import lasagne
from lasagne import layers
from lasagne import objectives
from chibi.module import export, import_
from chibi.file import Chibi_path

import theano
import theano.tensor as T

import numpy as np

__all__ = [ 'Network' ]


def get_function( name ):
    name = name.lower()
    if name == 'softmax':
        return lasagne.nonlinearities.softmax


class Layer( Chibi_object ):
    name = descriptor.String()

    def real_layer( self ):
        raise NotImplementedError

    @property
    def dict( self ):
        return dict( name=self.name, type=export( self ) )

    @property
    def yaml( self ):
        return yaml.dump( self.dict )

    @classmethod
    def from_dict( cls, d ):
        return cls( **d )


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


class Dense( Layer ):
    input = descriptor.String()
    function_no_liniarity = descriptor.String_choice(
        default=None, choice=( 'softmax', '', None ) )
    number_units = descriptor.Integer()

    def __init__( self, *args, **kw ):
        super().__init__( *args, **kw )
        self._function_no_liniarity = get_function(
            self.function_no_liniarity )

    @property
    def real_layer( self ):
        return layers.DenseLayer( self._input, num_units=self.number_units )

    @Layer.dict.getter
    def dict( self ):
        result = super().dict
        result.update( dict(
            input=self.input,
            function_no_liniarity=self.function_no_liniarity,
            number_units=self.number_units ) )
        return result


class Network( Chibi_object ):
    """
    Contendor de redes neuronales usando el framework lasagne

    Atributes
    ---------
    name: string
    """
    name = descriptor.String( default='test_network' )
    layers = descriptor.Dict()
    meta_layers = descriptor.Dict()
    order_layers = descriptor.List_kind_strict( kind=str )
    functions = descriptor.Dict()

    def __init__( self, layers=None, meta_layers=None, order_layers=None,
                  *args, **kargs ):
        if layers is None:
            layers = {}
        if meta_layers is None:
            meta_layers = {}
        if order_layers is None:
            order_layers = []
        super().__init__( layers=layers, meta_layers=meta_layers,
                          order_layers=order_layers, functions={},
                          *args, **kargs )

    def add_layer( self, layer ):
        self.layers[ layer.name ] = layer
        self.order_layers.append( layer.name )

    def save( self, path=None, folder='.' ):
        if path is None:
            path = self.name

        path = Chibi_path( path )
        if path.is_a_folder:
            path += self.name
        path = path.add_extensions( 'chibi', 'neuronal', 'yml' )

        layers = { name: layer.dict for name, layer in self.layers.items() }

        file_content = {
            'layers': layers,
            'order_layers': self.order_layers,
            'name': self.name,
        }

        path.open().write_yaml( file_content )
        return path

        mad_files.write_json( full_rute, file_content )

    @classmethod
    def load( cls, path ):
        path = Chibi_path( path )
        self = cls()
        read = path.open().read_yaml()
        for name, dict_layers in read[ 'layers' ].items():
            layer_type = import_( dict_layers[ 'type' ] )
            self.layers[ name ] = layer_type.from_dict( dict_layers )
        self.name = read[ 'name' ]
        self.order_layers = read[ 'order_layers' ]
        return self
