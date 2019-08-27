from chibi.object import Chibi_object, descriptor
from chibi.atlas import Chibi_atlas
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


class Layer( Chibi_object ):
    name = descriptor.String()

    @property
    def real_layer( self ):
        try:
            return self._real_layer
        except AttributeError:
            self.build_layer()
            return self._real_layer

    def build_layer( self ):
        raise NotImplementedError

    @property
    def dict( self ):
        return dict( name=self.name, type=export( self ) )

    @property
    def yaml( self ):
        return yaml.dump( self.dict )

    @classmethod
    def from_dict( cls, d, net=None ):
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

    def build_layer( self ):
        self._real_layer = layers.InputLayer(
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


class Input_field( descriptor.Kind ):
    kind = Input

class layer_field( descriptor.Kind ):
    kind = Layer


class Dense( Layer ):
    input = layer_field()
    function_no_liniarity = descriptor.String_choice(
        default=None, choice=( 'softmax', 'sigmoid', 'softplus', '', None ) )
    number_units = descriptor.Integer()

    def __init__( self, *args, **kw ):
        super().__init__( *args, **kw )

    @property
    def function( self ):
        if self.function_no_liniarity == "softmax":
            return lasagne.nonlinearities.softmax
        if self.function_no_liniarity == "sigmoid":
            return lasagne.nonlinearities.sigmoid
        if self.function_no_liniarity == "softplus":
            return lasagne.nonlinearities.softplus
        elif self.function_no_liniarity == "":
            return None
        else:
            raise NotImplementedError

    def build_layer( self ):
        self._real_layer = layers.DenseLayer(
            self.input.real_layer, num_units=self.number_units,
            nonlinearity=self.function )

    @Layer.dict.getter
    def dict( self ):
        result = super().dict
        result.update( dict(
            input=self.input.name,
            function_no_liniarity=self.function_no_liniarity,
            number_units=self.number_units ) )
        return result

    @classmethod
    def from_dict( cls, d, net ):
        input_layer = net.layers[ d.input ]
        d = d.copy()
        d[ "input" ] = input_layer
        return cls( **d )


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
        super().__init__(
            layers=layers, meta_layers=meta_layers,
            order_layers=order_layers, functions={}, *args, **kargs )

    def add_layer( self, layer ):
        self.layers[ layer.name ] = layer
        self.order_layers.append( layer.name )

    def save( self, path=None ):
        if path is None:
            path = self.name

        path = Chibi_path( path )
        if path.is_a_folder:
            path += self.name
        path_network = path.add_extensions( 'chibi', 'neuronal', 'yml' )
        path_params = path.add_extensions( 'chibi', 'neuronal', 'npz' )

        layers = { name: layer.dict for name, layer in self.layers.items() }

        file_content = {
            'layers': layers,
            'order_layers': self.order_layers,
            'name': self.name,
        }

        path_network.open().write_yaml( file_content )

        params = lasagne.layers.get_all_param_values( self.layer_output )
        np.savez( path_params, *params )

        return path_network

    @classmethod
    def load( cls, path ):
        path = Chibi_path( path )
        self = cls()
        read = path.open().read_yaml()
        self.name = read[ 'name' ]
        self.order_layers = read[ 'order_layers' ]

        for layer_name in self.order_layers:
            layer_dict = read.layers[ layer_name ]
            layer_type = import_( layer_dict[ 'type' ] )
            self.layers[ layer_name ] = layer_type.from_dict(
                layer_dict, self )

        path_params = path.replace_extensions( 'npz' )
        with np.load( path_params ) as f:
            param_values = [
                f[ 'arr_%d' % i ]
                for i in range( len( f.files ) ) ]
        lasagne.layers.set_all_param_values( self.layer_output, param_values )
        return self

    @property
    def layer_output( self ):
        layer = self.layers[ self.order_layers[-1] ]
        return layer.real_layer

    def build_functions( self, deterministic=False ):
        l_out = self.layer_output
        x_sym = T.lmatrix()
        y_sym = T.lvector()

        output = lasagne.layers.get_output(
            l_out, x_sym, deterministic=deterministic )
        pred = output.argmax( -1 )

        #loss = objectives.categorical_crossentropy( output, y_sym ).mean()
        loss = objectives.binary_crossentropy( output, y_sym ).mean()
        params = lasagne.layers.get_all_params( l_out )
        acc = T.mean( T.eq( output, y_sym ) )

        #grad = T.grad( loss, params )
        #updates = lasagne.updates.sgd( grad, params, learning_rate=0.01 )
        #updates = lasagne.updates.adam()
        updates = lasagne.updates.adam( loss, params )

        f_train = theano.function(
            [ x_sym, y_sym ], [ loss, acc ], updates=updates )
        f_train_pred = theano.function(
            [ x_sym, y_sym ], [ loss, acc, output ], updates=updates )
        f_val = theano.function( [ x_sym, y_sym ], [ loss, acc ] )

        f_predict = theano.function( [ x_sym ], pred )
        f_test_predict = theano.function( [ x_sym ], output )

        self.functions = Chibi_atlas( {
            'train': f_train,
            'train_predict': f_train_pred,
            'val': f_val,
            'predict': f_predict,
            'test_predict': f_test_predict,
        } )
        return self.functions
