from unittest import TestCase
from chibi.module import export
from chibi_genjuu.index import Population, Dweller, Sample
from elasticsearch_dsl import Document, field


from chibi_genjuu.network import Input, Network, Layer


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
        self.assertEqual( result_layer.dict == self.layer.dict )
