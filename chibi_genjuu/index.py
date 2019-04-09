from elasticsearch_dsl import Document, field, InnerDoc
from chibi.module import import_, export
from .dweller import Dweller
from .exceptions import Dangerous_purge


SAMPLE_NORMAL = 'normal'
SAMPLE_ALL    = 'all'
SAMPLE_TRAIN  = 'train'
SAMPLE_TEST   = 'test'


class Index_inner( InnerDoc ):
    index = field.Keyword()
    klass = field.Keyword()

    @property
    def map( self ):
        if self.klass:
            cls = import_( self.klass )
        else:
            cls = Dweller
        if self.index:
            cls._index._name = self.index
        return cls

    @map.setter
    def map( self, value ):
        self.klass = export( value )
        self.index = value._index._name

    @property
    def exists( self ):
        return self.map._index.exists()

    def purge( self ):
        if '*' in self.map._index._name:
            raise Dangerous_purge( self.map._index._name )
        self.map._index.delete()


class Dweller_inner( Index_inner ):
    pass


class Sample_inner( Index_inner ):
    name = field.Text( fields={ 'raw': field.Keyword(), } )
    description = field.Text()
    utilization = field.Keyword()


class Population( Document ):
    name = field.Text( fields={ 'raw': field.Keyword(), } )
    description = field.Text()
    dweller = field.Object( Dweller_inner )
    samples = field.Object( Dweller_inner, multi=True )

    class Index:
        name = "population"

    def add_sample( self, sample_class=None, index=None ):
        result = {}
        if sample_class is None:
            sample_class = Sample
        result[ 'klass' ] = export( sample_class )
        if index is not None:
            result[ 'index' ] = index
        self.samples.append( result )


class Sample( Document ):
    pass
