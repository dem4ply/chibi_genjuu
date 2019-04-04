from elasticsearch_dsl import Document, field, InnerDoc
from chibi.module import import_, export


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


class Dweller( Document ):
    pass


class Sample( Document ):
    pass
