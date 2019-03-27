import tempfile, shutil
from unittest import TestCase

from chibi.file import Chibi_file
from chibi.file.snippets import current_dir, cd

from chibi_genjuu.samples import (
    Sample, SAMPLE_STATUS_NORMAL, CURRENT_VERSION, Metric, Image_metric
)


class Test_samples( TestCase ):

    def setUp( self ):
        self.current_dir = current_dir()
        self.root_dir = tempfile.mkdtemp()
        cd( self.root_dir )

    def tearDown(self):
        cd( self.current_dir )
        shutil.rmtree( self.root_dir )

    def test_init( self ):
        sample = Sample()
        self.assertEqual( sample.name, 'no_name' )
        self.assertEqual( sample.description, '' )
        self.assertEqual( sample.utilization, SAMPLE_STATUS_NORMAL )
        self.assertEqual( sample.version, CURRENT_VERSION )
        self.assertEqual( sample.metrics, [] )
        return sample

    def test_save_should_create_a_yaml_file( self ):
        sample = Sample()
        sample.save()
        result = Chibi_file( '{}.yml'.format( sample.name ) ).read_yaml()
        self.assertEqual( result, sample.to_dict() )

    def test_save_with_metrics( self ):
        sample = Sample()
        sample.metrics.append( Metric( value='test', real=1 ) )
        sample.save()
        result = Chibi_file( '{}.yml'.format( sample.name ) ).read_yaml()
        metrics = result[ 'metrics' ]
        self.assertEqual( metrics[0].to_dict(), { 'value': 'test', 'real':1 } )

    def test_when_load_the_file_should_load_the_metrics( self ):
        sample = Sample( name='second_test' )
        sample.metrics.append( Metric( value='test', real=1 ) )
        sample.metrics.append( Metric( value='none', real=None ) )
        sample.save()

        sample_2 = Sample( path=sample._file )
        self.assertEqual( len( sample_2.metrics ), 2 )
        self.assertListEqual( sample.metrics, sample_2.metrics )

    def test_should_save_the_type_of_the_metric( self ):
        sample = Sample( name='second_test' )
        sample.metrics.append( Metric( value='test', real=1 ) )
        sample.metrics.append( Image_metric( value='none', real=None ) )
        sample.save()

        sample_2 = Sample( path=sample._file )
        self.assertEqual( len( sample_2.metrics ), 2 )
        self.assertIsInstance( sample_2.metrics[0], Metric )
        self.assertIsInstance( sample_2.metrics[1], Image_metric )
