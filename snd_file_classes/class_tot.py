from time import timezone
from numpy.core.fromnumeric import var
from pandas.core.frame import DataFrame
from scipy import stats
import pandas as pd
import math
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
import datetime



def get_sgf_from_tot_file( f ):

    # extract total only soundings from SGF file
    def get_tot_soundings_from_sgf( raw_sgf ):
        all_soundings = raw_sgf.split( '$\n' )
        tot_soundings = []
        
        for sounding in all_soundings:
            if len(sounding)>len(sounding.upper().replace("HM=24", "")):
                tot_soundings.append('$\n' + sounding)
        return tot_soundings

    # read file to string
    def read_file( file_path ):
        file_contents = ''

        try:
            with open( file_path, 'r') as open_file: # open-read-close
                file_contents = open_file.read()
        except Exception as e:
            print( e )
        return file_contents

    # wrapper to read file and get tot contents
    def read_tot( file_path, sounding_nr = 0 ):
        raw_contents = read_file( file_path )
        if raw_contents == '': return None
        tot_soundings = get_tot_soundings_from_sgf( raw_contents )[sounding_nr]
        return tot_soundings.strip()

    return read_tot( file_path=f )



# class to read, analyze and present *.tot data. latest addition includes
# the ability to add diagrams (classifications and/or calculated values)
class tot():
    def __init__( self, sgf_data, layers=[], comment="", mask=[], apply_corrections=True, calculate_additional_variables=False ):
        self.headers = self.head_to_df( sgf_data )
        self.data = self.tot_to_df( sgf_data )
        self.clean_tot()
        self.layers = layers
        self.mask = mask
        self.classifications = []
        self.name = self.headers['HK'].iloc[0]
        self.comment = comment

# additional functions to add value to raw data
        if apply_corrections:
            self.apply_corrections()
            self.calc_tot_pow()
        if calculate_additional_variables:
            self.calc_vars()

    def calc_vars( self ):
# for clf SVV 2016
        gamma = 8 #kN/m3
        area = (0.057/2)**2 * math.pi

# stadard deviation
        _, var_fdt_0_30 = self.get_f_dt_running_var( interval=0.3 )
        std_fdt_0_30 = np.sqrt(var_fdt_0_30)
        self.data['std_fdt-0_3'] = std_fdt_0_30

# smoothed normalized penetration pressure
        D = self.data['D']
        A = self.data['A']
        sigma_eff = gamma * D
        sigma_eff = sigma_eff.clip(lower=0.0001) # avoid div_zero

        q_n = A / (sigma_eff * area)
        self.data['q_ns-0_3'] = self.running_average( D=D, variable=q_n, interval=0.3 ) # s: "smoothed"

    def running_average( self, D, variable, interval=0.1 ):
        v_use = variable.to_numpy()

        D = D.to_numpy()
        D = D - D[0]
        n = np.argmax(D>interval)

        n_before = int((n-1)/2)
        n_after = int(n/2)


        ans = np.array([])        
        # cases where i < n/2 (window touches list start)
        for i in range(n_before):
            k = n - n_before + i
            ans = np.append( ans, np.sum( v_use[:k] ) / k )

        # i > n/2 && i < N-n/2
        main_avg = np.cumsum(v_use, dtype=float)
        main_avg[n:] = main_avg[n:] - main_avg[:-n]        
        ans = np.append( ans, (main_avg[n - 1:] / n) )

        # i > N-n/2 (window touches list end)
        for i in range(n_after):
            k = n - (i + 1)
            ans = np.append( ans, np.sum( v_use[-k:] ) / k )

        return ans

    def add_classifications( self, classifications ):
        for some_method in classifications:
            self.classifications.append( some_method )
    

    def add_to_data( self, colname='', data=[]):
        if len(self.data) == len(data):
            self.data[colname] = data
        else:
            print('column ' + colname + ' not added as number of rows did not match dataframe')


    def get_tot_data( self ):
        return self.data


    def get_name( self ):
        return self.name


    def get_comment( self ):
        return self.comment


    def get_stats( self ):
        print( self.get_name() )
        print( self.data.describe(include='all') )


    def get_f_dt_running_var( self, interval=0.1 ):
        df = self.data #everything
        D = df['D'] # depth
        vars = D * 0 # new 0-series

        increment = round(D.diff().mode()[0],3)
        if abs(increment-0.025) > 0.004: # 0.025-ish
            print('careful! increment calculated at ' + str(increment) + ' while the standard is 0.025')
            import winsound
            frequency = 2000  # Set Frequency To 2000 Hertz
            duration = 600  # Set Duration To 600 ms == .6 second
            winsound.Beep(frequency, duration)


        for index, depth in D.items():
            val_group = df[ ( df['D']>=(depth-interval)) & ( df['D']<=(depth+interval) ) ]
            vars.iat[index] = val_group['A'].var()

        return D.tolist(), vars.tolist()


    def head_to_df( self, tot_string ):
        header_name = ""
        header_value = ""

# extract header to a single line string
        header = tot_string.split('#\n')[0]
        header = header.replace('$\n','')
        header = header.replace('\n',',')
        header = header.replace(',,',',')

        if header[-1] == ',':
            header = header[:-1]

        headers = header.split(',')

        for h in headers:
            if ( len(h) - (len( h.replace('=','') )) > 1):
# more than 1 "=" in header
                s = h.split('=')

# add "extra" all pieces back to s[1]
                for i in range(2,len(s)):
                    s[1] = s[1] + "=" + s[i]

            else:
                s = h.split('=')

            header_name = header_name + s[0] + ','
            header_value = header_value + s[1] + ','

        header_name = header_name[:-1]
        header_value = header_value[:-1]

        csv = header_name + '\n' + header_value
        csv = StringIO(csv)
        df = pd.read_csv(csv)
        return df


    def set_layers( self, layers=[] ):
        self.layers = layers


    def apply_mask( self, mask=[] ):
# will not work properly unless layering & mask are within D
# also assumes that len(mask)==len(self.layers)
        data = self.data

        layers = self.layers # here used as mask
        self.mask = mask
        
        id_la = []
        id_ma = []

# selects depths closest to specified mask/layering (if tie: select first)
        for l,m in zip( layers, mask ):
            id_la.append( data.iloc[(data['D']-l).abs().argsort()[:2]].index.tolist()[0] ) 
            id_ma.append( data.iloc[(data['D']-m).abs().argsort()[:2]].index.tolist()[0] )
            # sometimes gives: "Warning: _AXIS_NAMES has been deprecated."

# add first/last depth index if either list is missing it
        last_id = data.tail(1).index[0]
        first_id = data.head(1).index[0]
        if id_la[0] != first_id or id_ma[0] != first_id:
            id_la.append( first_id )
            id_ma.append( first_id )
        if id_la[-1] != last_id or id_ma[-1] != last_id:
            id_la.append( last_id )
            id_ma.append( last_id )
        
# new depth profile (NaN-s)
        tmp_D = self.data['D'].multiply(np.nan)

# assign values at at specific depths
        for l, m in zip(id_la,id_ma):
            tmp_D.iloc[l] = self.data['D'].iloc[m]

# interpolate NaN-s
        tmp_D.interpolate(inplace=True)

# replace depth column
        self.data['D'] = tmp_D


    def get_D_F_DT( self ):
        D = self.data['D'].tolist()
        A = self.data['A'].tolist()
        return D, A
    

    def get_D_Power( self ):
        depth = self.data['D'].tolist()
        power = self.data['tot_pow'].tolist()
        return depth, power
    

    def calc_tot_pow( self ):
        # bedrock :  9999999 kN
        # hammer  : ~150-180 kN
        # flush   :  ~60-90 kN 
        # rotation:   ~30-35 kN
        # normal  :    ~0-30 kN
        #pre-corrected to penetration speed of 5cm/sek (3m/min)


        power = pd.Series(np.zeros(len(self.data['A'])), index=self.data.index) # new Series
        self.data['tot_pow'] = power # append Series to df

        A = self.data['A']
        AP = self.data['AP']
        AR = self.data['AR']
        AQ = self.data['AQ']
        
        offset = AQ*10 + AR*50 + AP*80

# offset A to create tot_pow & add to data
        power = ( A + offset )
        self.data['tot_pow'] = power



    def tot_to_df( self, tot_string ):
        block_sep = "£@$£@"
        tot_data = tot_string.split('#\n')[1]
        tot_data = tot_data.split('\n#$')[0]
        tot_data = tot_data.replace(',%', ',TID=')
        tot_data = tot_data.replace(',k=', ',K=')
        oneline_data = tot_data.replace('\n', ',')
        oneline_data = tot_data.replace('\n', ',')
        keys = []
        csv = ''

# grab all headers
        all_regs = oneline_data.split(',')
        for reg in all_regs:
            if '=' in reg: 
                keys.append(reg.split('=')[0])

        keys = list(dict.fromkeys(keys)) # removes doubles, preserves order

# change block seperator to something unique
        tot_data = block_sep + tot_data	
        for k in keys:
            tot_data = tot_data.replace(',' + str(k), block_sep + str(k))
        tot_data = tot_data.replace('\n' , block_sep + '\n' + block_sep)

# loop through each line and add to csv
        csv = ','.join(keys)
        data_lines = tot_data.split('\n')	
        for line in data_lines:
            csv = csv + '\n'
            i=0
            for k in keys:
                s = line.split(block_sep + str(k) + '=')			
                if ( len(s)>1 ):
                    if i>0:
                        csv = csv + ','
                    csv = csv + s[1].split(block_sep, 1)[0]
                else:
                    csv = csv + ','
                i = i + 1

        csv = StringIO(csv)

        df = pd.read_csv(csv)
        df.set_index('D')
        return df


    def clean_tot( self ):
        blow_threshold  = 5.0    # MPa hydraulic pressure
        flush_threshold = 0.1    # MPa hydraulic pressure
        rotation_threshold = 40  # rpm 

# creates or updates AP ( 0:OFF 1:ON ) hammering column based on hydraulic hammer pressure, AZ (MPa).
        if 'AZ' in self.data.columns: # not alwas present
            if ( 'AP' not in self.data.columns ) or ( self.data.AP.mean() == 0 or self.data.AP.mean() == 1 ): # not found or constant on/off
                self.data.loc[self.data.AZ <= blow_threshold, 'AP'] = 0
                self.data.loc[self.data.AZ > blow_threshold, 'AP'] = 1

# creates or updates AR ( 0:OFF 1:ON ) flushing column based on flushing pressure, I (MPa)
        if 'I' in self.data.columns:
            if ( 'AR' not in self.data.columns ) or ( self.data.AR.mean() == 0 or self.data.AR.mean() == 1 ): # not found or constant on/off
                self.data.loc[self.data.I <= flush_threshold, 'AR'] = 0
                self.data.loc[self.data.I > flush_threshold, 'AR'] = 1

# creates or updates AQ ( 0:OFF 1:ON ) increased rotation column based on rotation, R (rpm)
        if 'R' in self.data.columns:
            if ( 'AQ' not in self.data.columns ) or ( self.data.AQ.mean() == 0 or self.data.AQ.mean() == 1): # not found or constant on/off
                self.data.loc[self.data.R <= rotation_threshold, 'AQ'] = 0
                self.data.loc[self.data.R > rotation_threshold, 'AQ'] = 1
        
# backup original penetration rate (needed for classification methods)
        if 'B' in self.data.columns:
            self.data['B_org'] = self.data['B']

    def apply_corrections( self, remove_push_breaks=True, force_rate_correction=True, apply_normal_increments=True ):
        if remove_push_breaks:
# remove breaks by linear interpolation over masked intervals
            rate = self.data['B'] # Rate of penetration (mm/s)
            force = self.data['A'] # Push force

            pen_rate = rate.median()
            st_dev = rate.std()
            N_stdev = 3

# push break definition: rate < mean - ( 3x stdev )
            rate_mask = ( rate >= pen_rate - (N_stdev * st_dev) )

# make sure first value is not nan ( skip interpolation )
            if rate_mask.iat[0] == False:
                rate_mask.iat[0] = True
                rate.iat[0] = pen_rate

# replace masked vals with nan-s
            rate.where(rate_mask, inplace=True)
            force.where(rate_mask, inplace=True)

# interpolate nan-s
            rate.interpolate( inplace=True )
            force.interpolate( inplace=True )

# return values to df
            self.data['A'] = force
            self.data['B'] = rate


        if force_rate_correction: # correct push force to normalized
            
            force = self.data['A'] # Push force
            rate = self.data['B'] # Rate of penetration (mm/s)

# model parameters as function of rate
            a = 8.02000000e-6 * rate**3 - 1.2001e-3     * rate**2 + 7.680927e-2  * rate - 8.370417e-1
            b =             0 * rate**3 + 3.2598e-4     * rate**2 - 2.167286e-2  * rate + 1.52108915
            c =             0 * rate**3 - 1.01357e-3    * rate**2 + 1.2439644e-1 * rate + 2.0146433e-1
            d = 2.81007567e-6 * rate**3 - 4.85590747e-4 * rate**2 + 3.028811e-2  * rate + 3.487437e-1

# calculate correction & update
            k = d + ( a-d ) / ( 1 + ( force/c )**b )
            updated_force = force / k

# apply changes
            self.data['A'] = updated_force
            self.data['B'] = rate * 0 + 50 # correct to normal rate


        if apply_normal_increments:
# assume normal penetration rates and utilize depth
            D = self.data['D']
            increment = round(D.diff().mode()[0],3)

            if abs(increment-0.025) > 0.002: # 0.025-ish
                factor_to_standard = 0.025/increment
                freq = str(factor_to_standard) + 'S'
                norm_depth = D / increment # set increment as seconds

# create datetime and use as index
                pen_time = pd.to_datetime( norm_depth, unit='s' )
                pen_time.rename("pen_as_time", inplace=True)
                sounding_data = self.data
                sounding_data = sounding_data.set_index(pen_time)

# resample to desired interval
                resampled_data = sounding_data.resample( freq ).asfreq()
# add first/last registration
                resampled_data.iloc[0] = sounding_data.iloc[0] # start ok to overwrite
                
# append last registration if not present in df
                if resampled_data.iloc[-1]['D'] != sounding_data.iloc[-1]['D']:
                    resampled_data = resampled_data.append(sounding_data.iloc[-1])

#interpolate missing values
# numeric values (linear interpolation)
                interp_resampl = resampled_data.interpolate()
# non-numeric values with fill
                interp_resampl = interp_resampl.fillna(method='ffill')
            
# return manipulated df
                self.data = interp_resampl.reset_index( drop=True )

    def to_figure( 
                    self, 
                    filename=None, 
                    split_main_figure=True, 
                    color_figure=False, 
                    plot_layers=False, 
                    show_layer_analysis=True,
                    extras={}, # extra variable against depth
                    show_classifications = False
                   ):

# total sounding data
        D = self.data['D'] # Depth
        A = self.data['A'] # Push force
        B = self.data['B'] # Rate of penetration (mm/s)
        BB = ( 1 / B ) * 1000  # Rate of penetration (s/m)
        R = self.data['R'] # Rate of rotation ( not used )
        I = self.data['I'] # Flushing pressure
        AP = self.data['AP'] # Hammering
        AR = self.data['AR'] # Flushing
        AQ = self.data['AQ'] # Increased rotation
        K  = self.data['K'] # Comments ( not used )

# base definitions
        _margin = 0.1
        _alpha = 0.1
        eps = np.finfo(float).eps

# sizes of sub-plots ( % of drawn space in standard figure - ie. will add to over 100% because of margins and additional figures)
        _sizes = { # figure relative sizes in %
            'F_DT': 0.5, 
            'flushing': 0.03,
            'hammering': 0.03,
            'extras': 0.5, # will add up to > 100%
            'class': 0.15, # ditto            
            }

# calculate total figure size
        k = 1 # constant added height for legend/margins (link to font-size?)

# additional witdh factors
        extras_figure = ( len(extras)>0 ) * 1 # ( 0 or 1 )
        no_class = show_classifications * len( self.classifications ) # ( 0 to n ) 
        added_size = ( extras_figure*(_sizes['extras'] + 2*_margin)) + \
                     ( (no_class>0) * (_margin + no_class*(_sizes['class']+_margin)) )

# define total size
        fig_width = 8 * ( 1 + added_size )
        fig_height = (k + (D.iat[-1] // 5) * 5 ) # new increment each 5m

        _colors = {
            'push_force' : (237/255,28/255,46/255), # RGB:237/28/46
            'penetration_rate' : (93/255,184/255,46/255), # RGB:93/184/46
            'flush_pressure' : (0,142/255,194/255), # RGB:0/142/194
            'rotation' : (0,0,0),
            'flushing' : (0,0,0),
            'hammering' : (0,0,0)
        }

        if not color_figure: # force black
            _colors =  { x:(0,0,0) for x in _colors}

# calculate rate size and accumulated positions ( incl. margins )
        _sizes['rate'] = 1 - ( _sizes['F_DT'] + _sizes['flushing'] + _sizes['hammering'] )
        _sizes = {key: _sizes[key] * ( (1-(2*_margin))/(1+added_size)) for key in _sizes}
        
        _sizes['all'] = ( 1-2*_margin ) / ( 1+added_size )
        _margin = _margin / (1+added_size)


        if split_main_figure:
            _sizes['rotation'] = _sizes['F_DT'] / 4
        else:
            _sizes['rotation'] = _sizes['F_DT'] / 3

        _sizes['POS_1'] = _margin
        _sizes['POS_2'] = _sizes['POS_1'] + _sizes['rate']
        _sizes['POS_3'] = _sizes['POS_2'] + _sizes['flushing']
        _sizes['POS_4'] = _sizes['POS_3'] + _sizes['hammering']
        _sizes['POS_5'] = _sizes['POS_4'] + _sizes['F_DT'] / 2
        _sizes['POS_6'] = _sizes['POS_4'] + _sizes['F_DT'] - _sizes['rotation'] #1 - _margin - _sizes['rotation']
        _sizes['POS_7'] = _sizes['POS_4'] + _sizes['F_DT'] + 2 * _margin
        _sizes['POS_8'] = _sizes['POS_7'] + ( extras_figure>0 ) * (_sizes['extras'] + 2 * _margin)

# define figure bounds
        fig = plt.figure(figsize=(fig_width, fig_height))

        if 'HK' in self.headers.columns:
            fig.suptitle( self.name, x=_sizes['POS_3'], fontsize=16)
        
        ax0 = fig.add_axes( [_sizes['POS_1'], 0.1, _sizes['rate'], 0.85] ) # rate of penetration
        ax1 = fig.add_axes( [_sizes['POS_1']+eps, 0.1, _sizes['rate']-eps, 0.85] ) # flushing pressure

        if split_main_figure:
            ax4 = fig.add_axes( [_sizes['POS_4'], 0.1, _sizes['F_DT']/2, 0.85] ) # push force ( 0 - 10)
            ax5 = fig.add_axes( [_sizes['POS_5'], 0.1, _sizes['F_DT']/2, 0.85] ) # push force (10 - 30)
            ax6 = fig.add_axes( [_sizes['POS_6'], 0.1, _sizes['rotation'], 0.85] ) # increased rotation
        else:
            ax4 = fig.add_axes( [_sizes['POS_4'], 0.1, _sizes['F_DT'], 0.85] ) # push force (0 - 30)
            ax5 = None
            ax6 = fig.add_axes( [_sizes['POS_6'], 0.1, _sizes['rotation'], 0.85] ) # increased rotation

# draw layer & hammer/flush figures last for black borders
        if plot_layers:
            ax_ = fig.add_axes( [_sizes['POS_1'], 0.1, _sizes['all'], 0.85] ) # rate of penetration
        ax2 = fig.add_axes( [_sizes['POS_2'], 0.1, _sizes['flushing'], 0.85] )
        ax3 = fig.add_axes( [_sizes['POS_3'], 0.1, _sizes['hammering'], 0.85] )

# draw extras
        if extras_figure>0:
            ax7 = fig.add_axes( [_sizes['POS_7'], 0.1, _sizes['extras'], 0.85] ) # extras figure

        if no_class > 0:
            class_ax = []
            k = 0
            for i in range(no_class):
                class_ax.append(fig.add_axes( [_sizes['POS_8'] + i*( _sizes['class']+ _margin ), 0.1, _sizes['class'], 0.85] )) # increased rotation)
        


# populate figures
        if plot_layers:
            ax_ = get_layer_barplot( ax_, self.layers, y_vals=D,suppress_y_axis=True, suppress_x_axis=True )
        ax0 = get_line_subplot( ax0, x_vals=BB, y_vals=D, x_min=0, x_max=300, invert_xaxis=True, color=_colors['penetration_rate'], use_ticks=[ 100, 200, 300], alpha=_alpha )
        ax1 = get_line_subplot( ax1, x_vals=I, y_vals=D, x_min=0, x_max=3, invert_xaxis=False, color=_colors['flush_pressure'], suppress_y_axis=True, use_ticks=[0, 1, 2 ], label_padding=25, alpha=0.001 )
        ax2 = get_flush_hammer_figure( ax2 , x_vals=AR, y_vals=D,color=_colors['flushing'], alpha=_alpha )
        ax3 = get_flush_hammer_figure( ax3 , x_vals=AP, y_vals=D,color=_colors['hammering'], alpha=_alpha )
        if split_main_figure:
            ax4 = get_line_subplot( ax4, x_vals=A, y_vals=D, x_min=0, x_max=10, color=_colors['push_force'], suppress_y_axis=True, use_ticks=[5], alpha=_alpha )
            ax5 = get_line_subplot( ax5, x_vals=A, y_vals=D, x_min=10, x_max=30, color=_colors['push_force'], suppress_y_axis=True, use_ticks=[10, 20, 30], alpha=_alpha )
        else:
            ax4 = get_line_subplot( ax4, x_vals=A, y_vals=D, x_min=0, x_max=30, color=_colors['push_force'], suppress_y_axis=True, use_ticks=[ 5, 10, 15, 20, 25, 30], alpha=_alpha )
        ax6 = get_rot_figure( ax6 , x_vals=AQ, y_vals=D,color=_colors['rotation'] )

        
        if extras_figure>0:            
            for series in extras:
                ax7.plot( extras[series]['x'], extras[series]['d'], linestyle='solid', linewidth = 1, label=series)
            ax7 = set_ax_ybounds( ax7, D )
            ax7 = format_ax_axis( ax7, invert_yaxis=True, invert_xaxis=False, suppress_x_axis=False, suppress_y_axis=False )
            ax7 = set_ax_ygrid( ax7,interval=0.5 )
            ax7 = color_ax_boundary( ax7,'0.5' )
            ax7.legend()
            ax7.patch.set_alpha( _alpha )


        if no_class > 0:
            for ax_i, model_i in zip( class_ax, self.classifications ):
                classes = model_i['results']['class']
                class_colors = [ model_i['legend'][x]['color'] for x in classes ]
                #colors = model_i['results']['legend']
                ax_i = get_layer_barplot( 
                    ax_i, 
                    model_i['results']['depth'], 
                    y_vals=D,
                    title=model_i['name'],
                    suppress_y_axis=False, 
                    suppress_x_axis=True, 
                    colors=class_colors,
                    legend=model_i['legend'],
                    alpha_layers=1.0
                    )


# add labels to axes
        ax1.set_ylabel( ylabel='Dybde (m)',labelpad=10 ) 
        multicolor_label( ax1,('  Spyletrykk (MPa)','Bortid (s/m)'), (_colors['flush_pressure'],_colors['penetration_rate']),axis='x',size=10,weight='normal',rotation=0, bbox_to_anchor=(0, -0.1) )
        multicolor_label( ax2,('Spyling', ' '),(_colors['flushing'], _colors['flushing']),axis='x',size=10,weight='normal',rotation=90, bbox_to_anchor=(0, -0.05) )
        multicolor_label( ax3,('Slagboring', ' '),(_colors['hammering'],_colors['hammering']),axis='x',size=10,weight='normal',rotation=90, bbox_to_anchor=(0, -0.06) )
        multicolor_label( ax4,('Matekraft (kN)',' '),(_colors['push_force'],_colors['push_force']),axis='x',size=10,weight='normal',rotation=0, bbox_to_anchor=(.2, -0.1) )
        
        if show_layer_analysis:
            add_layer_analysis( self.layers, self.data, ax1, ax4, ax5 )

# show or save results
        if filename == None:
            plt.show()
        else:
            plt.savefig( filename,dpi=600, transparent=True )
            plt.close()


def get_layer_Fdt_analysis( df, depth_from, depth_to ):
    # D = df['D'] # Depth
    A = df['A'] # Push force
    B = df['B'] # Rate of penetration (mm/s)

    pen_rate = B.median()
    st_dev = B.std()
    N_stdev = 3
    window=1#0.2 # mean +- this%  included
    decimals = 2

# filter data from df
    depth_mask = (df['D'] >= depth_from) & (df['D'] <= depth_to)
    rate_mask = (df['B'] >= pen_rate - (N_stdev * st_dev)) & (df['B'] <= pen_rate + (N_stdev * st_dev))
    
    df = df[depth_mask] # depth range
    D_layer = df['D'] # depth of layer

    df = df[rate_mask] # only accepted rate range
    df = df[df['AP'] == 0] # no hammering
    df = df[df['AR'] == 0] # no flushing
    df = df[df['AQ'] == 0] # no rotation

    D_used = df['D'].to_numpy()
    A_used = df['A'].to_numpy()

    if D_used.size != 0:
        slope, intercept, r_value, p_value, std_err = stats.linregress(D_used, A_used)
        
        if False:
            fig = plt.figure()
            plt.plot(D_used,A_used)
            plt.plot(D_used, intercept + slope*D_used, 'r', label='fitted line')
            plt.axis('equal')
            
            xmin, _ = plt.xlim()
            ymin, _ = plt.ylim()

            plt.annotate('y=' + str(slope) + 'x + ' + str(intercept) + '\nr2=' + str(r_value**2) ,
                            xy=( xmin , ymin ),
                            xytext=(0, 0),
                            textcoords="offset points",
                            ha='left', va='bottom')
            plt.show()


        r_res = ''
        r_res += 'F = {:.{}f}'.format(slope, decimals) + 'z + {:.{}f}kN'.format(intercept, decimals) + '\n'
        r_res += 'r = {:.{}f}'.format(r_value**2, 4)



        
        l_res =''
        l_res += 'using: ' + '{:.{}f}'.format( (len(df.index)/len(D_layer))*100, 1 ) + '% (' + str(len(df.index)) + '/' + str(len(D_layer)) + ')\n'
        l_res+= 'rate: ' + '{:.{}f}'.format(df['B'].mean(), decimals) + 'mm/s. \u03C3:' + '{:.{}f}'.format(df['B'].std(), decimals) + '\n'
        l_res+= 'F_dt: ' + '{:.{}f}'.format(df['A'].mean(), decimals) + 'kN. \u03C3:' + '{:.{}f}'.format(df['A'].std(), decimals)
    else:
        l_res, r_res, [slope, intercept] = '', '', [0, 0]

    return l_res, r_res, [slope, intercept]


def add_layer_analysis( layers, data, ax1, ax4, ax5 ):
    v_offset = 2
    h_offset = 2

# loop through layers
    for i in range(len(layers)-1):
        analysis_text_left, analysis_text_right, line = get_layer_Fdt_analysis( df=data, depth_from=layers[i], depth_to=layers[i+1] )


# left annotation
        ax1.annotate(analysis_text_left,
                    xy=( 0 , layers[i]),
                    xytext=(h_offset, -v_offset),
                    textcoords="offset points",
                    ha='left', va='top')

# right annotation
        if ax5 is not None:
            ax5.annotate(analysis_text_right,
                        xy=( 30 , layers[i+1]),
                        xytext=(-h_offset, v_offset),
                        textcoords="offset points",
                        ha='right', va='bottom',color='r')
        else:
            ax4.annotate(analysis_text_right,
                        xy=( 30 , layers[i+1]),
                        xytext=(-5, +6),
                        textcoords="offset points",
                        ha='right', va='bottom',color='r')


# draw best fit line
        depth = [layers[i], layers[i+1]]
        F=[]
        for d in depth:
            F.append( d * line[0] + line[1])
        
        ax4.plot( F, depth, color='r', linestyle='--', linewidth=3)
        if ax5 is not None:
            ax5.plot( F, depth, color='r', linestyle='--', linewidth=3)

# use Gaussian Process Regressor ?
        D = data['D'].tolist() # Depth
        A = data['A'].tolist() # Push force
        #print(D)
        #print(A)



def get_layer_barplot( ax, layers, y_vals,x_min=0, colors=[], legend={}, title='', x_max=1, invert_yaxis=True, invert_xaxis=False, suppress_x_axis=True, suppress_y_axis=True, alpha=0.0, alpha_layers=0.2 ):
    ax = set_ax_ybounds(ax, y_vals)
    ax.set_xlim (x_min, x_max )

# settings
    ax = format_ax_axis(ax, invert_yaxis,invert_xaxis,suppress_x_axis,suppress_y_axis)
    ax.patch.set_alpha( alpha )

    bar_ind = [0.5]
    bar_width = [1]

    if layers !=[]:
        bars_list = []
        for i in range(len(layers)-1):
            depth_of_layer = layers[i]
            
            layer_thickness = layers[i+1] - layers[i]

            if not colors:
                bars_list.append( ax.bar(bar_ind, layer_thickness, width=bar_width, bottom=depth_of_layer, alpha=alpha_layers) )
            else:
                color_of_layer = colors[i]
                bars_list.append( ax.bar(bar_ind, layer_thickness, width=bar_width, color=color_of_layer, bottom=depth_of_layer, alpha=alpha_layers) )
 
    if legend:
        class_names =[]
        for key in legend:
            class_name = legend[key]['name']
            if class_name != '' and class_name not in class_names: # one instance + exclude no-class
                class_names.append( class_name )
                ax.bar( 0, 0, width=0, color=legend[key]['color'], label=class_name )

        ax.legend( prop={'size': 8} )
        
    
    if title !='':
        ax.title.set_text(title)
    return ax


def get_rot_figure( ax, x_vals, y_vals, color='black', linestyle='solid', linewidth = 1, invert_yaxis=True, invert_xaxis=False, suppress_x_axis=True, suppress_y_axis=True ):
# collapse states to layers
    depth_from, depth_to, state = layers_from_binary_parameter( x_vals.tolist(), y_vals.tolist() )

    _low = 0
    _high = 1

    ax.set_xlim (_low, _high )
    ax = set_ax_ybounds(ax, y_vals)

# settings
    ax = format_ax_axis(ax, invert_yaxis,invert_xaxis,suppress_x_axis,suppress_y_axis)
    ax.tick_params(axis=u'both', which=u'both',length=0, pad=0, labelcolor=color)
    ax.patch.set_alpha( 0.0 )

    for depth_f, depth_t, st in zip( depth_from, depth_to, state ):
        if st == 1: # draw an X
            x = [ _low, _high, _low, _high, _low ]
            y = [ depth_f, depth_f, depth_t, depth_t, depth_f ]
            ax.plot( x, y, color=color, linestyle=linestyle, linewidth=linewidth)
    
    ax = color_ax_boundary(ax,'0.5')

    return ax

def get_flush_hammer_figure( ax, x_vals, y_vals, color='black', linestyle='solid', linewidth = 1, invert_yaxis=True, invert_xaxis=False, suppress_x_axis=True, suppress_y_axis=True, alpha=1.0 ):
# collapse states to layers
    depth_from, depth_to, state = layers_from_binary_parameter( x_vals.tolist(), y_vals.tolist() )

    _low = 0
    _high = 1
    _vlines = 3

    ax.set_xlim (_low, _high )
    ax = set_ax_ybounds(ax, y_vals)

# settings
    ax = format_ax_axis(ax, invert_yaxis,invert_xaxis,suppress_x_axis,suppress_y_axis)
    ax.tick_params(axis=u'both', which=u'both',length=0, pad=0, labelcolor=color)
    ax.patch.set_alpha( alpha )
    
    for depth_f, depth_t, st in zip( depth_from, depth_to, state ):
        if st == 1:
# rectangle
            x = [ _low, _high, _high, _low, _low ]
            y = [ depth_f, depth_f, depth_t, depth_t, depth_f ]
            ax.plot( x, y, color=color, linestyle=linestyle, linewidth=linewidth)

# hatch
            n = _vlines + 1
            for i in range( n ):
                x_hatch = _low + (_high - _low) * (i+1) / n
                x = [ x_hatch, x_hatch ]
                y = [ depth_f, depth_t ]
                ax.plot( x, y, color=color, linestyle=linestyle, linewidth=linewidth)
    return ax


def layers_from_binary_parameter( x_vals, y_vals ):
    depth_from = []
    depth_to = []
    state = []
    
# first registration
    state.append( x_vals[0] )
    depth_from.append( y_vals[0] )
    
    x_temp = x_vals[0]
    for x, y in zip(x_vals, y_vals):
        if x != x_temp: # only keep changes
            state.append( x )
            depth_to.append( y )
            depth_from.append( y )
            x_temp = x   
    
    depth_to.append(y_vals[-1])
    return depth_from, depth_to, state


def get_line_subplot( ax, x_vals, y_vals, x_min=None, x_max=None, color='black', linestyle='solid', linewidth = 1, invert_yaxis=True, invert_xaxis=False, suppress_x_axis=False, suppress_y_axis=False, use_ticks=False, label_padding=10, alpha=1.0 ):
    ax.plot( x_vals, y_vals, color=color, linestyle=linestyle, linewidth=linewidth)#, marker=marker, markersize=marker_size)
    
    if ((x_min != None) and (x_max != None)):
        ax.set_xlim (x_min, x_max )
    ax = set_ax_ybounds(ax, y_vals)

# settings
    if use_ticks:
        ax.set_xticks( use_ticks, minor=False )
    ax = format_ax_axis( ax, invert_yaxis,invert_xaxis,suppress_x_axis,suppress_y_axis )
    ax.tick_params( axis=u'both', which=u'both',length=0, pad=label_padding, labelcolor=color )
    ax.tick_params(axis='y', colors='black')
    ax = set_ax_ygrid( ax,interval=0.5 )
    ax = color_ax_boundary( ax,'0.5' )

    ax.patch.set_alpha( alpha )
    return ax

def color_ax_boundary( ax, color ):
    for spine in ax.spines.values():
        spine.set_edgecolor( color )
    return ax

def set_ax_ygrid( ax, interval=5 ):
    start, end = ax.get_ylim()
    rng = np.arange( min(start, end),max(start, end), interval )
    ax.yaxis.set_ticks( rng )
    ax.grid( True )
    return ax

def set_ax_ybounds( ax, y_vals ):
    y_min = 0
    y_max = (( y_vals.iat[-1] // 5 ) + 1 ) * 5
    ax.set_ylim( y_min, y_max )
    return ax

def format_ax_axis(ax, invert_yaxis,invert_xaxis,suppress_x_axis,suppress_y_axis):
    if invert_yaxis:
        ax.invert_yaxis()
    if invert_xaxis:
        ax.invert_xaxis()
    if suppress_x_axis:
        ax.axes.get_xaxis().set_ticklabels([])
        ax.set_xticks([])
    if suppress_y_axis:
        ax.axes.get_yaxis().set_ticklabels([])
    return ax

# inspiration:https://stackoverflow.com/questions/33159134/matplotlib-y-axis-label-with-multiple-colors
def multicolor_label(ax,list_of_strings,list_of_colors,axis='x',anchorpad=0,bbox_to_anchor=(0, 0),**kw):
    from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

    # x-axis label
    if axis=='x' or axis=='both':
        boxes = [TextArea(text, textprops=dict(color=color, ha='left',va='bottom',**kw)) 
                    for text,color in zip(list_of_strings,list_of_colors) ]
        xbox = HPacker(children=boxes,align="center",pad=0, sep=5)
        anchored_xbox = AnchoredOffsetbox(loc='center left', child=xbox, pad=anchorpad,frameon=False,bbox_to_anchor=bbox_to_anchor,
                                          bbox_transform=ax.transAxes, borderpad=0.)
        ax.add_artist(anchored_xbox)

    # y-axis label
    if axis=='y' or axis=='both':
        boxes = [TextArea(text, textprops=dict(color=color, ha='left',va='center',**kw)) 
                     for text,color in zip(list_of_strings[::-1],list_of_colors) ]
        ybox = VPacker(children=boxes,align='center', pad=0, sep=5)
        anchored_ybox = AnchoredOffsetbox(loc='center left', child=ybox, pad=anchorpad, frameon=False, bbox_to_anchor=bbox_to_anchor, 
                                          bbox_transform=ax.transAxes, borderpad=0.)
        ax.add_artist(anchored_ybox)


def get_scatter_subplot( ax, x_vals, y_vals, x_min, x_max, labels, marker, marker_size, invert_yaxis=True, invert_xaxis=False, suppress_x_axis=False, suppress_y_axis=False, use_ticks=False, alpha=1.0 ):
    y_min = 0
    y_max = (( y_vals.iat[-1] // 5 ) + 1 ) * 5

    ax.scatter( x_vals, y_vals, c=labels, s=marker_size, marker=marker )
    ax.set_ylim( y_min, y_max )
    ax.set_xlim (x_min, x_max )

# settings
    ax = format_ax_axis(ax, invert_yaxis,invert_xaxis,suppress_x_axis,suppress_y_axis)
    ax.patch.set_alpha( alpha )

    ax = set_ax_ygrid(ax,interval=5)
    return ax