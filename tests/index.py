from unittest import TestCase
from chibi.module import export
from chibi_genjuu.index import Population, Dweller, Sample
from elasticsearch_dsl import Document, field


class Dweller_test( Document ):
    pass


class Sample_test( Document ):
    pass


class Test_population( TestCase ):
    def setUp( self ):
        self.original_index = Population._index._name
        Population._index._name = "test__{}".format( self.original_index )
        self.population = Population()

    def tearDown( self ):
        Population._index._name = self.original_index

    def test_if_no_have_dwaller_class_should_get_the_default( self ):
        self.population.dweller.index = "test__dweller"
        dweller = self.population.dweller.map
        self.assertEqual( dweller, Dweller )
        self.assertEqual( dweller._index._name, "test__dweller" )

    def test_when_have_dwaller_class_should_get_the_expected_class( self ):
        self.population.dweller.index = "test__dweller"
        self.population.dweller.klass = export( Dweller_test )
        dweller = self.population.dweller.map
        self.assertEqual( dweller, Dweller_test )
        self.assertEqual( dweller._index._name, "test__dweller" )

    def test_when_add_a_new_sample_should_use_default( self ):
        self.assertFalse( self.population.samples )
        self.population.add_sample()
        self.assertTrue( self.population.samples )
        self.assertEqual( self.population.samples[0].map, Sample )

    def test_when_add_a_new_sample_using_a_class_should_be_saved( self ):
        self.assertFalse( self.population.samples )
        self.population.add_sample( Sample_test )
        self.assertTrue( self.population.samples )
        self.assertEqual( self.population.samples[0].map, Sample_test )

    def test_should_save_the_custom_index_of_the_sample( self ):
        self.assertFalse( self.population.samples )
        self.population.add_sample( Sample_test, index="asdf" )
        self.assertTrue( self.population.samples )
        self.assertEqual( self.population.samples[0].map._index._name, "asdf" )

    def test_should_save_the_custom_index_of_the_sample_2( self ):
        self.assertFalse( self.population.samples )
        self.population.add_sample( index="asdf" )
        self.assertTrue( self.population.samples )
        self.assertEqual( self.population.samples[0].map._index._name, "asdf" )
