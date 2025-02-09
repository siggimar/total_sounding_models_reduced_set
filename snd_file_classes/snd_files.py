import os
import re
import utm
import snd_file_classes.snd_methods as snd_methods


class snd_file():
    def __init__( self, file_path ): # changes to init: delete .pkl!
        self.file_path = file_path
        self.min_data_threshold = 2
        self.has_sounding_data = False
        
        self.soundings = {
            'cpt': [], # cone penetration tesst
            'tot': [], # total sounding
            'rps': []  # rotary pressure sounding
        }
        self.read_raw_data( file_path )


    def has_data( self ):
        used_keys = []
        for k in self.soundings.keys():
            if self.soundings[k]:
                used_keys.append( k )
        return used_keys


    def has_method( self, method ):
        methods = self.has_data()
        return method in methods


    def read_raw_data( self, file_path ):
        self.raw_data = ''
        if os.path.isfile( file_path ):
            with open( file_path, 'r' ) as f:
                self.raw_data = f.read()


    def parse_raw_data( self ):
        self.get_coords()
        self.get_id()

        self.read_sounding_data()

        n = len(self.raw_data) - len( self.raw_data.replace('*\n','') )

        if n>8:
            a=1
        a=1
        if self.has_sounding_data==False:
            a=1


    def add_lowess( self, delta ):
        for method in self.soundings:
            if self.soundings[method]: 
                for sounding in self.soundings[method]:
                    sounding.add_lowess( delta )
                    #if method=='cpt': sounding.to_figure()

    def add_qn( self, delta ):
        for method in self.soundings:
            if self.soundings[method]: 
                for sounding in self.soundings[method]:
                    sounding.add_qn()
                    sounding.add_qns( delta )


    def read_sounding_data( self ):
        self.is_cpt = False

        blocks = self.raw_data.split('*\n')
        pattern = r'\b\d{1,3}\s\d{1,2}\.\d{1,2}\.\d{4}\b'

        for block in blocks:
            match = re.match(pattern, block)
            if match:
                if self.block_has_sounding_data( block ):
                    self.has_sounding_data = True
                    method_class, method_str = self.set_method( block )
                    if method_class is not None: 
                        self.soundings[method_str].append( method_class )


    def block_has_sounding_data( self, block ):
        lines = block.strip().split('\n')
        lines = [ l.strip() for l in lines ]

        #print(lines[0] + '\n' + lines[1])
        if len(lines)>2:
            d_0 = float( lines[2].split(' ')[0] )
            d_1 = float( lines[-1].split(' ')[0] )
            dist = d_1-d_0
            if dist > self.min_data_threshold: return True
            else:
                #print(dist)
                a=1
        return False


    def set_method( self, block ):
        method = int(block.split(' ')[0])
        
        if method==7:
            a=1
            self.is_cpt = True
            return snd_methods.snd_cpt( block, self ), 'cpt'
        elif method==23: # see: grain_size_data\gsa_dataset\17 Finnmark\A32.SND
            return snd_methods.snd_rps( block, self ), 'rps'
        elif method==24:
            a=1 # not in dataset
        elif method==25:
            return snd_methods.snd_tot( block, self ), 'tot'
        else:
            pass # 21:108, 22:22, 26:23
        return None, '' # no data or no method matched


    def get_coords( self ):
        lines = self.raw_data.split('*\n')[0].strip().split('\n')
        
        try:
            x = float( lines[1] ) if lines[1].replace('.','').replace('-','').isnumeric() else 0
            y = float( lines[0] ) if lines[0].replace('.','').replace('-','').isnumeric() else 0
            z = float( lines[2] ) if lines[2].replace('.','').replace('-','').isnumeric() else 0

            self.utm_coords = { # borehole coordinates, z value at terrain
                'x': min( x, y ), # easting
                'y': max( x, y ), # northing
                'z': z
            }

        except Exception as e:
            print(self.file_path)
            print(e)


        if False:
            lat_lng = utm.to_latlon(self.utm_coords['x'],self.utm_coords['y'], 33, northern=True)
            self.lat_lng_coords = {
                'x': lat_lng[1], # east
                'y': lat_lng[0], # north
            }


    def get_id( self ):
        try:
            lines = self.raw_data.split('*\n99')[1].split('*\n')[0].strip().split('\n')
            self.project_name = lines[2]
            self.pos_name = lines[3]
        except Exception as e: # priblem with some files in 'GuDb_data' folder
            self.project_name = ''
            self.pos_name = self.file_path.split('\\')[-1]
        
        self.pos_name = self.pos_name.replace('.SND','')

            #print(self.file_path)
            #print(e)
