import snd_file_classes.method_plotter as method_plotter
import numpy as np
from snd_file_classes.lowess import lowess as lowess
import re


class snd_method_base():
    def __init__( self, block, reference ):
        self.reference = reference # parent position
        
        lines = block.split('\n')

        self.method_nr = lines[0].strip().split( ' ' )[0]
        self.date = lines[0].strip().split( ' ' )[1]

        header_data = lines[1].split(' GUID ')[0].split(' ')
        self.raw_file = header_data[-1].strip()
        self.stop_code = header_data[1].strip()

        self.raw_data = '\n'.join(lines[2:])

        # replace consecutive spaces with single space
        multiple_space_pattern = r' +'
        self.raw_data = re.sub( multiple_space_pattern, ' ', self.raw_data )

        self.lowess_data = {}
        self.qn_data = {}

        self.update()


    def update( self ):
        self.parse_raw_data()
        self.scale_data()
        self.discard_raw_data()


    def discard_raw_data( self ):
        self.raw_data = None


    def set_code_flags( self, codes ):
        
        if ' 70 ' in codes or 'r1' in codes:
            self.increased_rotation = True
        elif ' 71 ' in codes or 'r2'in codes: # overlooked if ' 70 ' or 'r1' is in codes
            self.increased_rotation = False

        if ' 72 ' in codes or 'y1' in codes:
            self.flushing = True
        elif ' 73 ' in codes or 'y2'in codes: # similar
            self.flushing = False

        if ' 74 ' in codes or 's1' in codes:
            self.hammering = True
        elif ' 75 ' in codes or 's2'in codes: # ...
            self.hammering = False

        for r_code in [ ' 40 ', ' 41 ', ' 43 ', ' 44 ', ' 45 ', ' 46 ']:
            if r_code in codes:
                self.rock_drilling = True
        if ' 42 ' in codes: # drill-through ( even paced ) - overwrites rock drilling
            self.rock_drilling = False


    def parse_raw_data( self ):
        # set at start, codes seldom given
        self.increased_rotation = False
        self.flushing = False
        self.hammering = False
        self.rock_drilling = False


    def add_lowess( self, delta ):
        id = 'lowess_' + str( int(delta*100) ) # cm

        smoother = lowess( delta=delta )
        smoother.fit( self.data[0], self.data[1] )

        self.lowess_data[id] = smoother.predict()


    def add_qn( self ):
        area = (0.057/2)**2 * np.pi
        gamma = 7 # (17-10)kN/m3
        sigma_eff = gamma * np.clip(self.data[0] , a_min=0.001, a_max=None ) # avoid div_zero
        self.qn_data['q_n'] = self.data[1] / (sigma_eff * area)


    def add_qns( self, delta ):
        self.qn_data['q_ns'] = self.running_average( self.data[0], self.qn_data['q_n'], delta )


    def running_average( self, x, y, interval=0.1 ):
        fact = 100 # cm_accuracy
        diffs = np.round(np.diff( x*fact ), 0)
        diffs = diffs[(diffs !=0)] # double registrations sometimes cause problems
        
        delta = np.argmax( np.bincount( diffs.astype(int) ) )/fact
        eps = delta/50
        if delta == 0:
            a=1
        
        

        x_us = np.arange( min(x), max(x)+eps, delta )
        y_us = np.interp( x_us, x, y )
        
        x_usn = x_us - x_us[0]
        n = np.argmax( x_usn>interval )

        n_before = int( (n-1)/2 )
        ans = np.array( [] )

        for i in range( n_before ): # cases where i < n/2 (window touches list start)
            k = n - n_before + i
            ans = np.append( ans, np.sum( y_us[:k] ) / k )

        # i > n/2 && i < N-n/2
        main_avg = np.cumsum( y_us, dtype=float )
        main_avg[n:] = main_avg[n:] - main_avg[:-n]
        ans = np.append( ans, (main_avg[n - 1:] / n) )

        for i in range( int(n/2) ): # i > N-n/2 (window touches list end)
            k = n - (i + 1)
            ans = np.append( ans, np.sum( y_us[-k:] ) / k )
        return np.interp( x, x_us, ans )
    def scale_data( self ):
        pass


class snd_tot( snd_method_base ):
    def __init__( self, block, reference ):
        super().__init__( block, reference )


    def parse_raw_data( self ):
        super().parse_raw_data()

        raw_lines = self.raw_data.split('\n')
        dt = np.dtype('i4') # integer

        self.data = [                   # index, description, ( SGF-code:file unit {::scaled unit} )
            np.array( [] ),             # 0:depth ( D:m )
            np.array( [] ),             # 1:f_dt ( A:kPa::MPa )
            np.array( [] ),             # 2:flushing ( I:kPa )
            np.array( [] ),             # 3:rate ( s/dm::s/m )
            np.array( [], dtype=dt ),   # 4:increased_rotation ( 1/0 )
            np.array( [], dtype=dt ),   # 5:Flushing ( 1/0 )
            np.array( [], dtype=dt ),   # 6:Hammering ( 1/0 )
            np.array( [], dtype=dt )    # 7:assumed_rock_drilling ( 1/0 )
            ]

        for raw_line in raw_lines:
            line_data = raw_line.strip()

            n=line_data.count(' ')
            if n < 3: continue # not enough blocks

            blocks = line_data.split( ' ' )

            for j in range(4):
                self.data[j] = np.append( self.data[j], float(blocks[j]) )

            rest = ''
            if n>3:
                rest = blocks[4:]
                rest = ' ' + ' '.join(rest).strip().lower() + ' '

            self.add_coded_columns( rest )


    def add_coded_columns( self, codes): # columns
        self.set_code_flags( codes )

        self.data[4] = np.append( self.data[4], int(self.increased_rotation) )
        self.data[5] = np.append( self.data[5], int(self.flushing) )
        self.data[6] = np.append( self.data[6], int(self.hammering) )
        self.data[7] = np.append( self.data[7], int(self.rock_drilling) )


    def scale_data( self ):
        self.data[1] /= 1000 # kPa to MPa
        self.data[3] *= 10 # s/dm



    def to_figure( self, multiple_soundings=False ):
        plotter = method_plotter.tot_plotter( self )

        if multiple_soundings: 
            return plotter.plot( multiple_soundings )
        plotter.plot()




class snd_rps( snd_method_base ):
    def __init__( self, block, reference ):
        super().__init__( block, reference )


    def parse_raw_data( self ):
        super().parse_raw_data()

        raw_lines = self.raw_data.split('\n')
        dt = np.dtype('i4') # integer

        self.data = [ 
            np.array( [] ),             # 0:depth (D:m)
            np.array( [] ),             # 1:f_dt (A:kPa)
            np.array( [], dtype=dt ),   # 2:increased_rotation (1/0)
            np.array( [], dtype=dt )    # 3:assumed_rock_drilling (1/0)
            ]

        for raw_line in raw_lines:
            line_data = raw_line.strip()

            n=line_data.count(' ')
            if n < 1: continue # not enough blocks

            blocks = line_data.split( ' ' )

            for j in range(2):
                self.data[j] = np.append( self.data[j], float(blocks[j]) )

            rest = ''
            if n>1:
                rest = blocks[2:]
                rest = ' ' + ' '.join(rest).strip().lower() + ' '

            self.add_coded_columns( rest )


    def add_coded_columns( self, codes): # columns
        self.set_code_flags( codes )

        self.data[2] = np.append( self.data[2], int(self.increased_rotation) )
        self.data[3] = np.append( self.data[3], int(self.rock_drilling) ) # codes can indicate rockface/penetration


    def scale_data( self ):
        self.data[1] /= 1000


    def to_figure( self, multiple_soundings=False ):
        plotter = method_plotter.rps_plotter( self )

        if multiple_soundings: 
            return plotter.plot( multiple_soundings )
        plotter.plot()




class snd_cpt( snd_method_base ):
    """a very basic CPTu class, just measured data - no in-situ stresses"""
    def __init__( self, block, reference ):
        super().__init__( block, reference )


    def parse_raw_data( self ):
        super().parse_raw_data()

        raw_lines = self.raw_data.split('\n')

        self.data = [ 
            np.array( [] ), # 0:depth (D:m)
            np.array( [] ), # 1:q_c (QC:kPa)
            np.array( [] ), # 2:u_2 (U:kPa)
            np.array( [] ), # 3:f_s (FS:kPa)
            #np.array( [], dtype=np.dtype('i4') ),   # 4:some coded column? (1/0)
            ]

        for raw_line in raw_lines:
            line_data = raw_line.strip()

            n=line_data.count(' ')
            if n < 1: continue # not enough blocks

            blocks = line_data.split( ' ' )

            for j in range(4):
                self.data[j] = np.append( self.data[j], float(blocks[j]) )

            rest = ''
            if n>1:
                rest = blocks[2:]
                rest = ' ' + ' '.join(rest).strip().lower() + ' '


    def to_figure( self, multiple_soundings=False ):
        plotter = method_plotter.cpt_plotter( self )
        
        if multiple_soundings: 
            return plotter.plot( multiple_soundings )
        plotter.plot()