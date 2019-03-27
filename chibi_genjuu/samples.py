import logging

from chibi.file import Chibi_file
from chibi.file.snippets import exists
from chibi.object import Chibi_object
from chibi.object.descriptor import (
    String, List_kind_strict, Integer, Descriptor, String_choice
)


logger = logging.getLogger( 'chibi_genjuu.samples' )


__all__ = [ 'Sample', 'Metric', 'Image_metric' ]


CURRENT_VERSION = 1


SAMPLE_STATUS_NORMAL = 'normal'
SAMPLE_STATUS_ALL    = 'all'
SAMPLE_STATUS_TRAIN  = 'train'
SAMPLE_STATUS_TEST   = 'test'
SAMPLE_STATUS = (
    SAMPLE_STATUS_ALL, SAMPLE_STATUS_NORMAL,
    SAMPLE_STATUS_TEST, SAMPLE_STATUS_TRAIN )


class Metric( Chibi_object ):
    """
    modelo de metricas para las muestras
    """
    real = Descriptor()
    value = Descriptor()

    def to_dict( self ):
        return {
            'real': self.real,
            'value': self.value,
        }

    def __eq__( self, other ):
        if isinstance( other, Metric ):
            return self.value == other.value and self.real == other.real
        return False


class Image_metric( Metric ):
    def __init__( self, *args, **kw ):
        super().__init__( *args, **kw )
        if not exists( self.value ):
            logger.warning( "cannot find the file '{}'".format( self.value ) )


class Sample( Chibi_object ):
    name = String( default='no_name' )
    description = String( default='' )
    metrics = List_kind_strict( kind=Metric )
    utilization = String_choice(
        default=SAMPLE_STATUS_NORMAL, choice=SAMPLE_STATUS )
    version = Integer( default=CURRENT_VERSION )

    def __init__(
            self, path=None, metrics_in_ram=True, metrics=None, *args, **kw ):
        if metrics is None:
            metrics = []
        super().__init__( *args, metrics=metrics, **kw )
        self.metrics_in_ram = metrics_in_ram
        if isinstance( path, str ):
            self._file = Chibi_file( path )
            self.refresh_from_db()
        elif isinstance( path, Chibi_file ):
            self._file = path
            self.refresh_from_db()
        else:
            self._file = None

    def refresh_from_db( self ):
        result = self._file.read_yaml()
        self.name = result[ 'name' ]
        self.description = result[ 'description' ]
        if self.metrics_in_ram:
            self.metrics = result[ 'metrics' ]
        self.utilization = result[ 'utilization' ]
        self.version = result[ 'version' ]

    def to_dict( self ):
        metrics = [ m.to_dict() for m in self.metrics ]
        return dict(
            name=self.name, description=self.description,
            metrics=self.metrics, utilization=self.utilization,
            version=self.version )

    def save( self ):
        if self._file is None:
            self._file = Chibi_file( "{}.yml".format( self.name ) )
        self._file.write_yaml( self.to_dict() )
