import logging
import os

from chibi.file.snippets import exists
from chibi.file.image import Chibi_image
from elasticsearch_dsl import Document, field


logger = logging.getLogger( 'chibi_genjuu.dweller' )


class Dweller( Document ):
    value_raw = field.Keyword()

    def __init__( self, *args, value=None, **kw ):
        super().__init__( *args, **kw )
        self.value = value

    @property
    def value( self ):
        return self.value_raw

    @value.setter
    def value( self, value ):
        self.value_raw = str( value )


class Image( Dweller ):
    def __init__( self, *args, value, **kw ):
        super().__init__( *args, value=value, **kw )
        if not self.value.exists:
            logger.warning( "cannot find the file '{}'".format(
                self.value._file_name ) )

    @Dweller.value.getter
    def value( self ):
        return Chibi_image( self.value_raw )

    @value.setter
    def value( self, value ):
        if isinstance( value, str ):
            value = Chibi_image( value )
        self.value_raw = value._file_name
        self.mime = value.properties.mime
        self.extension = value.properties.extension
        self.file = value.file_name
        self.dir = value.dir
        self.album = os.path.split( self.dir )[-1]
