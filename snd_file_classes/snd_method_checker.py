import numpy as np
import snd_file_classes.snd_methods as snd_m


class method_checker():
    def __init__(self) -> None:
        self.min_method_increment = { 'tot':0.03, 'rps':0.03, 'cpt':0.02}
        self.desired_method_increment = { 'tot':0.025, 'rps':0.025, 'cpt':0.02}


    def sounding_ok( self, snd_method ):
        if isinstance( snd_method, snd_m.snd_tot ):
            return self.tot_check( snd_method )
        if isinstance( snd_method, snd_m.snd_rps ):
            return self.rps_check( snd_method )
        if isinstance( snd_method, snd_m.snd_cpt ):
            return self.cpt_check( snd_method )


    def calc_most_common_increment( self, d ):
        diff_mm = np.diff(d * 1000).astype(int) # increments in mm
        diff_mm = diff_mm[ diff_mm>0 ] # fix problem with double recordings
        return np.argmax(np.bincount( diff_mm ))/1000 # most common increment in m


    def interval_ok( self, d, method ):
        logging_interval = self.calc_most_common_increment( d )
        if logging_interval>self.min_method_increment[method]:
            return False
        return True


    def tot_check( self, snd ):
        method = 'tot'
        d = snd.data[0]
        if not self.interval_ok( d, method ): return False
        # add more short circuit tests
        return True # no short circuits


    def rps_check( self, snd ):
        return True
    def cpt_check( self, snd ):
        return True