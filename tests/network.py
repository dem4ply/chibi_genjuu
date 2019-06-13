from unittest import TestCase
import tempfile, shutil
from chibi.module import export
from chibi_genjuu.index import Population, Dweller, Sample
from elasticsearch_dsl import Document, field
import lasagne


from chibi_genjuu.network import Input, Network, Layer, Dense

class Test_with_folder( TestCase ):
    def setUp(self):
        super().setUp()
        self.root_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree( self.root_dir )
        return super().tearDown()




class Test_layer( TestCase ):
    def setUp( self ):
        super().setUp()
        self.layer = Layer( name="layer_test" )


class Test_layer_input( Test_layer ):
    def setUp( self ):
        super().setUp()
        self.layer = Input( name="input" )

    def test_by_default_use_long_for_input_var( self ):
        self.assertEqual( self.layer.type_input_var, 'int64' )

    def test_can_parse_int_for_use_in_the_input_var( self ):
        self.layer = Input( name="input", input_var='int32' )
        self.assertEqual( self.layer.type_input_var, 'int32' )

    def test_real_shape( self ):
        real_layer = self.layer.real_layer
        self.assertEqual( real_layer.shape, ( None, None ) )
        self.assertEqual( real_layer.name, self.layer.name )

    def test_from_dict( self ):
        d = self.layer.dict
        result_layer = Input.from_dict( d )
        self.assertEqual( result_layer.dict, self.layer.dict )


class Test_layer_dense( Test_layer ):
    def setUp( self ):
        super().setUp()
        self.layer = Dense( name="dense", input='input' )

    def test_from_dict( self ):
        d = self.layer.dict
        result_layer = Dense.from_dict( d )
        self.assertEqual( result_layer.dict, self.layer.dict )

    def test_if_function_no_liniarity_is_none_should_be_create( self ):
        self.assertIsNone( self.layer._function_no_liniarity )

    def test_if_function_no_liniarity_is_none_should_be_create( self ):
        self.layer = Dense(
            name="dense", input='input',
            function_no_liniarity='softmax' )

        self.assertEqual(
            lasagne.nonlinearities.softmax,
            self.layer._function_no_liniarity )


class Test_network( Test_with_folder ):
    def setUp( self ):
        super().setUp()
        self.net = Network( name=self._testMethodName )

    def test_init( self ):
        self.assertEqual( self.net.name, "test_init" )
        self.assertIsInstance( self.net.layers, dict )
        self.assertEqual( self.net.layers, dict() )
        self.assertIsInstance( self.net.meta_layers, dict )
        self.assertEqual( self.net.meta_layers, dict() )
        self.assertIsInstance( self.net.functions, dict )
        self.assertEqual( self.net.functions, dict() )
        self.assertIsInstance( self.net.order_layers, list )
        self.assertEqual( self.net.order_layers, list() )

    def test_add_layer_input( self ):
        self.assertFalse( self.net.layers, "la red tiene capas" )
        layer = Input( name="input" )
        self.net.add_layer( layer )
        self.assertEqual( layer, self.net.layers[ layer.name ] )

    def test_save( self ):
        self.assertFalse( self.net.layers, "la red tiene capas" )
        layer = Input( name="input" )
        self.net.add_layer( layer )
        path = self.net.save( self.root_dir )
        result = path.open().read_yaml()
        expected = {
            'name': 'test_save',
            'order_layers': [ 'input' ],
            'layers': { 'input': {
                'batch_size': None,
                'input_var': 'int64',
                'name': 'input',
                'name': 'input',
                'shape': [ None ],
                'type': 'chibi_genjuu.network.Input',
            } }
        }
        self.assertEqual( expected, result )


class Test_network_save_load( Test_with_folder ):
    def setUp( self ):
        super().setUp()
        self.net = Network( name=self._testMethodName )
        self.assertFalse( self.net.layers, "la red tiene capas" )
        self.layer = Input( name="input" )
        self.net.add_layer( self.layer )

    def test_save( self ):
        path = self.net.save( self.root_dir )
        result = path.open().read_yaml()
        expected = {
            'name': 'test_save',
            'order_layers': [ 'input' ],
            'layers': { 'input': {
                'batch_size': None,
                'input_var': 'int64',
                'name': 'input',
                'name': 'input',
                'shape': [ None ],
                'type': 'chibi_genjuu.network.Input',
            } }
        }
        self.assertEqual( expected, result )

    def test_load( self ):
        path = self.net.save( self.root_dir )
        new_network = Network.load( path )
        self.assertEqual( self.net.order_layers, new_network.order_layers )
        self.assertEqual( self.net.name, new_network.name )
        for layer_name in self.net.order_layers:
            self.assertEqual(
                self.net.layers[ layer_name ].dict,
                new_network.layers[ layer_name ].dict, )
