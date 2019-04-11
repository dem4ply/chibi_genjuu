import logging
import os

from chibi.file.snippets import exists
from chibi.file.image import Chibi_image, Chibi_path
from elasticsearch_dsl import Document, field
from chibi.file.snippets import join, add_extensions, mkdir


logger = logging.getLogger( 'chibi_genjuu.dweller' )


class Dweller( Document ):
    value_raw = field.Keyword()
    real_raw = field.Float()

    def __init__( self, *args, value=None, **kw ):
        super().__init__( *args, **kw )
        if value is not None:
            self.value = value

    @property
    def value( self ):
        return self.value_raw

    @value.setter
    def value( self, value ):
        self.value_raw = str( value )

    @property
    def real( self ):
        return self.real_raw


class Image( Dweller ):
    mime = field.Keyword()
    extension = field.Keyword()
    file = field.Keyword()
    dir = field.Keyword()
    album = field.Keyword()
    base_path = field.Keyword()
    thumbnail_dir = field.Keyword()
    thumbnail_path = field.Keyword()

    def __init__( self, *args, **kw ):
        super().__init__( *args, **kw )
        if self.value_raw and not self.value.exists:
            logger.warning( "cannot find the file '{}'".format(
                self.value.path ) )

    @Dweller.value.getter
    def value( self ):
        return Chibi_image( self.value_raw )

    @value.setter
    def value( self, value ):
        if value is None:
            value = self.value_raw

        if isinstance( value, str ):
            value = Chibi_image( value )

        self.value_raw = value.path
        self.mime = value.properties.mime
        self.extension = value.properties.extension
        self.file = value.file_name
        self.dir = value.dir
        self.album = os.path.split( self.dir )[-1]

        self.base_path = list( os.path.split( self.dir ) )
        self.base_path.pop()
        self.base_path = join( *self.base_path )

        if self.thumbnail_dir:
            self.thumbnail_path = Chibi_path( self.thumbnail_dir )
        else:
            self.thumbnail_path = add_extensions(
                self.base_path, "thumbnail" )
        self.thumbnail_path = self.thumbnail_path + self.album

        mkdir( self.thumbnail_path )
        thumbnail = value.thumbnail( self.thumbnail_path )
        self.thumbnail_path = thumbnail.path

    @property
    def thumbnail( self ):
        return Chibi_image( self.thumbnail_path )
