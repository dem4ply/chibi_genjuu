from unittest import TestCase
from chibi.module import export
from chibi_genjuu.index import Population, Sample
from chibi_genjuu.dweller import Dweller
from elasticsearch_dsl import Document, field


class Dweller_test( Dweller ):
    @Dweller.value.getter
    def value( self ):
        return self.value_raw[::-1]


class Test_dweller( TestCase ):
    def test_dweller_return_a_string_in_value( self ):
        dweller = Dweller( value="hola" )
        self.assertEqual( dweller.value, "hola" )


class Test_dweller_test( TestCase ):
    def test_dweller_return_a_string_in_value( self ):
        dweller = Dweller_test( value="hola" )
        self.assertEqual( dweller.value, "aloh" )
