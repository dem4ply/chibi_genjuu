class Dangerous_purge( Exception ):
    def __init__( self, index_purge, *args, **kw ):
        msg = "the index containt a '*' index: '{}' ".format( index_purge )
        super().__init__( msg, *args, **kw )
