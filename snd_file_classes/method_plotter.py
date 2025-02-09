import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import numpy as np

class base_method_plotter():
    def __init__( self, reference ):
        self.reference = reference
        self.data = self.reference.data

        self.d_max, self.d_min = self.depth_range( self.data[0][-1], 0 )

        self.colors = {
            'f_dt' : (237/255,28/255,46/255), # RGB:237/28/46 - red
            'rate' : (93/255,184/255,46/255), # RGB:93/184/46 - green
            'flush_pressure' : (0,142/255,194/255), # RGB:0/142/194 - blue
            'rotation' : (0,0,0),
            'rock' : (0,0,0),
            'flushing' : (0,0,0),
            'hammering' : (0,0,0),
            'method_symbol' : ( 0,0,0 ),
            'q_c' : (68/255,79/255,85/255), # RGB:68/79/85 - dark grey
            'f_s' : (68/255,79/255,85/255), # RGB:237/28/46
            'u_2' : (0,142/255,194/255), # RGB:237/28/46
        }

        self.sizes = { # figure relative sizes in %
            'margin': 0.11,
            'f_dt': 0.58,
            'rate': 0.35,
            'flushing': 0.03,
            'hammering': 0.03,
            'extras': 0.5, # will add up to > 100%
            'class': 0.15, # ditto
            'q_c': 0.6,
            'u_2': 0.6,
            'f_s': 0.2
        }

    def generate_sizes_dict( self, width_vars, figure_blocks ):
        res = {}

        var_width_sum = 0

        for v in width_vars:
            res[v] = self.sizes[v]
            var_width_sum += res[v] # sum of all elements ( ex margins )
        desired_sum = 1 - ( 1+figure_blocks ) * self.sizes['margin'] # desired size

        # scale plot_items
        res = { k: v * desired_sum / var_width_sum for k,v in res.items() }
        res['margin'] = self.sizes['margin'] # add margin

        return res

    def show( self ):
        plt.show()


    def draw_curve( self, ax, x_vals, y_vals, x_min=None, x_max=None, color='black', linestyle='solid', \
                      linewidth = 1, invert_yaxis=True, invert_xaxis=False, suppress_x_axis=False, \
                      suppress_y_axis=False, use_ticks=False, label_padding=10, alpha=0.0 ):

        ax.plot( x_vals, y_vals, color=color, linestyle=linestyle, linewidth=linewidth)#, marker=marker, markersize=marker_size)

        if ((x_min != None) and (x_max != None)):
            ax.set_xlim (x_min, x_max )
        self.set_ax_ybounds(ax, y_vals[-1])

    # settings
        if use_ticks:
            ax.set_xticks( use_ticks, minor=False )

        plt.minorticks_on() # for y-ticks
        self.format_ax_axis( ax, invert_yaxis,invert_xaxis,suppress_x_axis,suppress_y_axis )
        ax.tick_params( axis=u'both', which=u'both',length=0, pad=label_padding, labelcolor=color )
        ax.tick_params( axis='y', colors='black' )

        self.set_ax_ygrid( ax,interval=5 )
        self.color_ax_boundary( ax, '0.5' )

        ax.patch.set_alpha( alpha )


    def rot_figure( self, ax, x_vals, y_vals, color='black', linestyle='solid', linewidth = 1, \
                        invert_yaxis=True, invert_xaxis=False, suppress_x_axis=True, suppress_y_axis=True ):
    # collapse states to layers
        depth_from, depth_to, state = self.layers_from_binary_parameter( x_vals, y_vals )

        x_low = 20
        x_high = 30

        ax.set_xlim (20, 30 )
        self.set_ax_ybounds( ax, y_vals[-1] )

    # settings
        self.format_ax_axis( ax, invert_yaxis,invert_xaxis,suppress_x_axis,suppress_y_axis )
        ax.tick_params( axis=u'both', which=u'both',length=0, pad=0, labelcolor=color )
        ax.patch.set_alpha( 0.0 )

        for depth_f, depth_t, st in zip( depth_from, depth_to, state ):
            if st == 1: # draw an X
                x = [ x_low, x_high, x_low, x_high, x_low ]
                y = [ depth_f, depth_f, depth_t, depth_t, depth_f ]
                ax.plot( x, y, color=color, linestyle=linestyle, linewidth=linewidth)

        ax = self.color_ax_boundary( ax, '0.5' )


    def draw_rock( self, ax, x_vals, y_vals,color='black', linestyle='solid', linewidth = 1, \
                        invert_yaxis=True, invert_xaxis=False, suppress_x_axis=True, suppress_y_axis=True ):
        depth_from, depth_to, state = self.layers_from_binary_parameter( x_vals, y_vals )

        if self.reference.stop_code.strip() in ['93', '94']: # add rock if only indicator is stop-code
            if not state[-1] == 1:
                depth_from.append( y_vals[-1] )
                depth_to.append( y_vals[-1] )
                state.append( 1 )

        ax.axis('off')

        x_low = 0
        x_high = 1
        y_margin = 0.08
        n_x = 5

        box_width = ( x_high-x_low ) / n_x
        dx = box_width/4
        dy = dx*2
        self.set_ax_ybounds( ax, y_vals[-1] )

    # settings
        self.format_ax_axis( ax, invert_yaxis,invert_xaxis,suppress_x_axis,suppress_y_axis, siltent=True )
        ax.tick_params( axis=u'both', which=u'both',length=0, pad=0, labelcolor=color )
        ax.patch.set_alpha( 0.0 )

        j = 0
        for depth_f, depth_t, st in zip( depth_from, depth_to, state ):
            j += 1
            if st == 1: # line with x-es
                x = [ x_low, x_high ]
                y = [ depth_f, depth_f ]
                ax.plot( x, y, color=color, linestyle=linestyle, linewidth=linewidth)

                for i in range( n_x ):
                    offset = i * box_width + box_width/2
                    x = [ offset-dx, offset+dx]
                    y_1 = [ depth_f + y_margin, depth_f + y_margin + dy ]
                    y_2 = [ depth_f + y_margin + dy, depth_f + y_margin ]
                    ax.plot( x, y_1, color=color, linestyle=linestyle, linewidth=linewidth)
                    ax.plot( x, y_2, color=color, linestyle=linestyle, linewidth=linewidth)

                if j<len(state):
                    x = [ x_low, x_high, x_high, x_low,x_low]
                    y = [ depth_f, depth_f, depth_t, depth_t, depth_f]
                    ax.plot( x, y, color=color, linestyle=linestyle, linewidth=linewidth)


        ax = self.color_ax_boundary( ax, '0.5' )


    def flush_hammer_figure( self, ax, x_vals, y_vals, color='black', linestyle='solid', linewidth = 1, \
                             invert_yaxis=True, invert_xaxis=False, suppress_x_axis=True, suppress_y_axis=True, alpha=0.0 ):
    # collapse states to layers
        depth_from, depth_to, state = self.layers_from_binary_parameter( x_vals, y_vals )

        _low = 0
        _high = 1
        _vlines = 3

        ax.set_xlim (_low, _high )
        self.set_ax_ybounds(ax, y_vals[-1])

    # settings
        self.format_ax_axis(ax, invert_yaxis,invert_xaxis,suppress_x_axis,suppress_y_axis)
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


    def layers_from_binary_parameter( self, x_vals, y_vals ):
        depth_from = []
        depth_to = []
        state = [] # list of x-values ( 0 or 1 )
        
        # first registration
        state.append( x_vals[0] )
        depth_from.append( y_vals[0] )

        temp_state = x_vals[0]
        for some_state, depth in zip(x_vals, y_vals):
            if some_state!=temp_state: # only keep changes
                state.append( some_state )
                depth_to.append( depth )
                depth_from.append( depth )
                temp_state = some_state   

        depth_to.append(y_vals[-1])
        return depth_from, depth_to, state


    def depth_range( self, y_max, y_min=0 ):
        plot_y_min = y_min # nothing clever here
        plt_y_max = ( ( y_max // 5 ) + 1 ) * 5
        return plt_y_max, plot_y_min


    def set_ax_ybounds( self, ax, y_max, y_min=0 ):
        y_min = 0
        y_max = ( ( y_max // 5 ) + 1 ) * 5
        ax.set_ylim( y_min, y_max )


    def format_ax_axis( self, ax, invert_yaxis,invert_xaxis,suppress_x_axis,suppress_y_axis, siltent=False):
        if invert_yaxis:
            ax.invert_yaxis()
        if invert_xaxis:
            ax.invert_xaxis()
        if suppress_x_axis:
            ax.axes.get_xaxis().set_ticklabels( [] )
            ax.set_xticks( [] )
        if suppress_y_axis:
            ax.axes.get_yaxis().set_ticklabels( [] )

        # restore coordinates in plot window
        if not siltent: ax.format_coord = lambda x, y: 'x={:g}, y={:g}'.format(x, y)


    def color_ax_boundary( self, ax, color ):
        for spine in ax.spines.values():
            spine.set_edgecolor( color )


    def set_ax_ygrid( self, ax, interval=5 ):
        start, end = ax.get_ylim()

        # style major/minor gridlines
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=0.7)
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black', alpha=0.3)

        # define intervals
        major_ticks = np.arange( min(start, end),max(start, end), interval )
        minor_ticks = np.arange( min(start, end),max(start, end), 1 )

        ax.minorticks_on()
        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)
        ax.xaxis.grid(False, which='minor')


    # inspiration:https://stackoverflow.com/questions/33159134/matplotlib-y-axis-label-with-multiple-colors
    def multicolor_label( self, ax,list_of_strings,list_of_colors,axis='x',anchorpad=0,bbox_to_anchor=(0, 0),**kw ):
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




class tot_plotter( base_method_plotter ):
    def __init__( self, data, split_main_figure=True, w=None, h=None ):
        super().__init__( data )
        self.split_main_figure = split_main_figure

        # figure element sizes and positions
        self.sizes_plot = self.generate_sizes_dict( ['f_dt', 'rate', 'flushing', 'hammering'], 1 ) #

        self.sizes_plot['rotation'] = self.sizes_plot['f_dt'] / 4
        if not split_main_figure: self.sizes_plot['rotation'] = self.sizes_plot['f_dt'] / 3

        # calculate subplot x-positions


        self.sizes_plot['p1'] = self.sizes_plot['margin'] # figure block (flushing & rate)
        self.sizes_plot['p2'] = self.sizes_plot['p1'] + self.sizes_plot['rate'] # flushing
        self.sizes_plot['p3'] = self.sizes_plot['p2'] + self.sizes_plot['flushing']# hammering
        self.sizes_plot['p4'] = self.sizes_plot['p3'] + self.sizes_plot['hammering']
        self.sizes_plot['p5'] = self.sizes_plot['p4'] + self.sizes_plot['f_dt'] / 2 # split main figure
        self.sizes_plot['p6'] = self.sizes_plot['p4'] + \
                                        self.sizes_plot['f_dt'] - \
                                        self.sizes_plot['rotation'] # rotation
        self.sizes_plot['p0'] = self.sizes_plot['p3'] - self.sizes_plot['margin'] * 0.5 # method symbol

        # define total size
        k = 5
        self.fig_width = 8 * ( 1 + 0 ) if w is None else w
        self.fig_height = ( 1 + (max(self.data[0][-1],5) // k) * k ) if h is None else h # new increment each 5m


    def draw_method_symbol( self, ax ):
        ax.axis('off')

        # line around circle
        thetas = np.linspace(0, 2 * np.pi, 100)

        r = 5
        x = r * np.cos( thetas ) # circle coordinates
        y = r * np.sin( thetas )

        theta_0 = 20 * np.pi/180
        x_0 = r * np.cos( theta_0 )
        y_0 = r * np.sin( theta_0 )

        #  add T (lift pen with nans)
        x = np.append( x, np.array([ np.nan, x_0, -x_0, np.nan, 0, 0 ]) )
        y = np.append( y, np.array([ np.nan, y_0, y_0, np.nan, y_0, -r ]) )

        ax.plot( x, y, c=self.colors['method_symbol'] )

        ax.set_xlim (-6, 6 )
        ax.set_ylim (-6, 6 )
        ax.axis('equal')


    def plot( self, multiple_soundings=False, f_name='' ):
        fig = plt.figure( figsize=(self.fig_width, self.fig_height) )

        # show borhole name if given
        if not multiple_soundings or True:  
            fig.suptitle( self.reference.reference.pos_name, x=self.sizes_plot['p2'], fontsize=16)

        n = self.fig_width/self.fig_height/0.75

        axm = fig.add_axes( [self.sizes_plot['p0']+.05, 0.96, self.sizes_plot['margin']*0.8, 0.04*n] ) # push force ( 0 - 10)

        ax1 = fig.add_axes( [self.sizes_plot['p1'], 0.1, self.sizes_plot['rate'], 0.85] ) # flushing
        ax2 = fig.add_axes( [self.sizes_plot['p1'], 0.1, self.sizes_plot['rate'], 0.85] ) # rate
        ax3 = fig.add_axes( [self.sizes_plot['p2'], 0.1, self.sizes_plot['flushing'], 0.85] ) # flushing
        ax4 = fig.add_axes( [self.sizes_plot['p3'], 0.1, self.sizes_plot['hammering'], 0.85] ) # hammering

        if self.split_main_figure:            
            ax5 = fig.add_axes( [self.sizes_plot['p4'], 0.1, self.sizes_plot['f_dt']/2, 0.85] ) # push force ( 0 - 10) 1
            ax6 = fig.add_axes( [self.sizes_plot['p5'], 0.1, self.sizes_plot['f_dt']/2, 0.85] ) # push force (10 - 30) 2
            ax7 = fig.add_axes( [self.sizes_plot['p6'], 0.1, self.sizes_plot['rotation'], 0.85] ) # increased rotation 3

            ax6.sharey(ax5)
            
            #ax5.get_shared_y_axes().join(ax5, ax6) # link y axis - split main figure
            
        else:
            ax5 = fig.add_axes( [self.sizes_plot['p4'], 0.1, self.sizes_plot['f_dt'], 0.85] ) # push force (0 - 30) 1
            ax6 = None
            ax7 = fig.add_axes( [self.sizes_plot['p6'], 0.1, self.sizes_plot['rotation'], 0.85] ) # increased rotation 3
        ax0 = fig.add_axes( [self.sizes_plot['p0'], 0.1, self.sizes_plot['margin']*0.8, 0.85] ) # rock symbols

        ax0.sharey(ax5)# link y axis
        ax1.sharey(ax5)
        ax2.sharey(ax5)
        ax3.sharey(ax5)
        ax4.sharey(ax5)
        ax7.sharey(ax5)
        
        self.draw_method_symbol( axm )
        if self.split_main_figure:
            self.draw_curve( ax5, x_vals=self.data[1], y_vals=self.data[0], x_min=0, x_max=10, \
                               color=self.colors['f_dt'], suppress_y_axis=True, use_ticks=[ 0, 5 ] ) # push force
            self.draw_curve( ax6, x_vals=self.data[1], y_vals=self.data[0], x_min=10, x_max=30, \
                               color=self.colors['f_dt'], suppress_y_axis=True, use_ticks=[10, 20, 30] )
            if hasattr( self.reference, 'lowess_data'):
                for lowess_interval in self.reference.lowess_data:
                    ax5.plot( self.reference.lowess_data[lowess_interval], self.data[0] )
                    ax6.plot( self.reference.lowess_data[lowess_interval], self.data[0] )
            
        else:
            self.draw_curve( ax5, x_vals=self.data[1], y_vals=self.data[0], x_min=0, x_max=30,\
                               color=self.colors['f_dt'], suppress_y_axis=True, use_ticks=[ 0, 5, 10, 15, 20, 25, 30] )
            if hasattr( self.reference, 'lowess_data'):
                for lowess_interval in self.reference.lowess_data:
                    ax5.plot( self.reference.lowess_data[lowess_interval], self.data[0] )
                    ax6.plot( self.reference.lowess_data[lowess_interval], self.data[0] )
        self.rot_figure( ax7 , x_vals=self.data[4], y_vals=self.data[0],color=self.colors['rotation'] )  # rotation

        self.draw_rock( ax0, x_vals=self.data[7], y_vals=self.data[0],color=self.colors['rock'] )  # rock drilling
        
        self.draw_curve( ax1, x_vals=self.data[2]/1000, y_vals=self.data[0], x_min=0, x_max=5, label_padding=25, \
                               color=self.colors['flush_pressure'], suppress_y_axis=False, use_ticks=[ 0, 1, 2, 3, 4 ] )  # flushing pressure
        self.draw_curve( ax2, x_vals=self.data[3], y_vals=self.data[0], x_min=0, x_max=500, invert_xaxis=True, \
                               color=self.colors['rate'], suppress_y_axis=True, use_ticks=[ 100, 200, 300, 400, 500]  ) # drill rate

        self.flush_hammer_figure( ax3, x_vals=self.data[5], y_vals=self.data[0],color=self.colors['flushing'] )
        self.flush_hammer_figure( ax4, x_vals=self.data[6], y_vals=self.data[0],color=self.colors['hammering'] )

# 2:flushing (I:kPa)
# 3:rate (s/dm)
# 5:Flushing (1/0)
# 6:Hammering (1/0)

        ax1.set_ylabel( ylabel='Depth (m)',labelpad=0 )
        self.multicolor_label( ax5,('Penetration force (kN)',' '),(self.colors['f_dt'],self.colors['f_dt']),\
                              axis='x',size=10,weight='normal',rotation=0, bbox_to_anchor=(.3, -0.08) )
        self.multicolor_label( ax1,('Penetration time (s/m)',' '), (self.colors['rate'],\
                        self.colors['rate']),axis='x',size=10,weight='normal',rotation=0, bbox_to_anchor=(-.3, -0.09) )
        self.multicolor_label( ax1,('Flushing pressure (MPa)',' '), (self.colors['flush_pressure'],\
                              self.colors['flush_pressure']),axis='x',size=10,weight='normal',rotation=0, bbox_to_anchor=(-.3, -0.11) )

        
#        self.multicolor_label( ax1,(' Flushing pressure (MPa)','Penetration time (s/m)'), (self.colors['flush_pressure'],\
#                        self.colors['rate']),axis='x',size=10,weight='normal',rotation=0, bbox_to_anchor=(0, -0.1) )


        self.multicolor_label( ax3,('Flushing', ' '),(self.colors['flushing'], self.colors['flushing']),\
                              axis='x',size=10,weight='normal',rotation=90, bbox_to_anchor=(0, -0.05) )
        self.multicolor_label( ax4,('Hammering', ' '),(self.colors['hammering'],self.colors['hammering']),\
                              axis='x',size=10,weight='normal',rotation=90, bbox_to_anchor=(0, -0.06) )

        #if len(str(f_name)) > 0: plt.savefig(str(f_name) + '.png', dpi=150)
        
        self.figure = fig
        self.axs = [ axm, ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7 ]

        
        if not multiple_soundings: plt.show()
        else: return ax1, ax2, ax3, ax4, ax5, ax6




class rps_plotter( base_method_plotter ):
    def __init__( self, data, split_main_figure=True ):
        super().__init__( data )
        self.split_main_figure = split_main_figure

        # figure element sizes and positions
        self.sizes_plot = self.generate_sizes_dict( ['f_dt'], 1 ) #

        self.sizes_plot['rotation'] = self.sizes_plot['f_dt'] / 4
        if not split_main_figure: self.sizes_plot['rotation'] = self.sizes_plot['f_dt'] / 3

        # calculate subplot x-positions
        self.sizes_plot['p0'] = self.sizes_plot['margin'] * 0.5 + .012
        self.sizes_plot['p1'] = self.sizes_plot['margin'] # main figure
        self.sizes_plot['p2'] = self.sizes_plot['p1'] + self.sizes_plot['f_dt'] / 2 # split main figure
        self.sizes_plot['p3'] = self.sizes_plot['p1'] + \
                                        self.sizes_plot['f_dt'] - \
                                        self.sizes_plot['rotation'] # increased rotation

        # define total size
        k = 5
        self.fig_width = 8 * ( 1 + 0 ) / 1.8
        self.fig_height = ( 1 + (max(self.data[0][-1],5) // k) * k ) # new increment each 5m
        


    def scale_data( self ):
        self.data[1] /= 1000


    def draw_method_symbol( self, ax ):
        ax.axis('off')

        # line around circle
        thetas = np.linspace(0, np.pi, 100)

        r = 5
        x = r * np.cos( thetas )
        y = r * np.sin( thetas )

        x = np.append( x, 0 )
        x = np.append( x, x[0] )

        y = np.append( y, -9 )
        y = np.append( y, -0 )
        ax.plot( x, y, c=self.colors['method_symbol'] )

        # fill
        ax.add_artist( Wedge( (0,0), r, 0, 180, fc=(0,0,0) ) )
        
        ax.set_xlim (-9, 9 )
        ax.set_ylim (-9, 5 )
        ax.axis('equal')


    def plot( self, multiple_soundings=False ):
        fig = plt.figure( figsize=(self.fig_width, self.fig_height) )

        # show borhole name if given
        if not multiple_soundings:  
            fig.suptitle( self.reference.reference.pos_name, x=self.sizes_plot['p2'], fontsize=16)

        n = self.fig_width/self.fig_height/0.75

        # define axis for each figure component
        axm = fig.add_axes( [self.sizes_plot['p0'], 0.96, self.sizes_plot['margin']*0.8, 0.04*n] ) # method figure
        ax0 = fig.add_axes( [self.sizes_plot['p0'], 0.1, self.sizes_plot['margin']*0.8, 0.85] ) # rock symbol
        if self.split_main_figure:            
            ax1 = fig.add_axes( [self.sizes_plot['p1'], 0.1, self.sizes_plot['f_dt']/2, 0.85] ) # push force ( 0 - 10)
            ax2 = fig.add_axes( [self.sizes_plot['p2'], 0.1, self.sizes_plot['f_dt']/2, 0.85] ) # push force (10 - 30)
            ax1.get_shared_y_axes().join(ax1, ax2)
            ax1.get_shared_y_axes().join(ax1, ax0)
        else:
            ax1 = fig.add_axes( [self.sizes_plot['p1'], 0.1, self.sizes_plot['f_dt'], 0.85] ) # push force (0 - 30)
            ax2 = None
        ax3 = fig.add_axes( [self.sizes_plot['p3'], 0.1, self.sizes_plot['rotation'], 0.85] ) # increased rotation
        ax1.get_shared_y_axes().join(ax1, ax3)

        # draw figure components
        self.draw_method_symbol( axm )
        if self.split_main_figure:
            self.draw_curve( ax1, x_vals=self.data[1], y_vals=self.data[0], x_min=0, x_max=10, \
                               color=self.colors['f_dt'], suppress_y_axis=False, use_ticks=[ 0, 5 ] )
            self.draw_curve( ax2, x_vals=self.data[1], y_vals=self.data[0], x_min=10, x_max=30, \
                               color=self.colors['f_dt'], suppress_y_axis=True, use_ticks=[10, 20, 30] )

            if hasattr( self.reference, 'lowess_data'):
                for lowess_interval in self.reference.lowess_data:
                    ax1.plot( self.reference.lowess_data[lowess_interval], self.data[0] )
                    ax2.plot( self.reference.lowess_data[lowess_interval], self.data[0] )
        else:
            self.draw_curve( ax1, x_vals=self.data[1], y_vals=self.data[0], x_min=0, x_max=30,\
                               color=self.colors['f_dt'], suppress_y_axis=False, use_ticks=[ 0, 5, 10, 15, 20, 25, 30] )
        self.rot_figure( ax3 , x_vals=self.data[2], y_vals=self.data[0],color=self.colors['rotation'] )

        self.draw_rock( ax0, x_vals=self.data[3], y_vals=self.data[0],color=self.colors['rock'] )

        ax1.set_ylabel( ylabel='Depth (m)',labelpad=0 )
        self.multicolor_label( ax1,('Matekraft (kN)',' '),(self.colors['f_dt'],self.colors['f_dt']),\
                              axis='x',size=10,weight='normal',rotation=0, bbox_to_anchor=(.65, -0.09) )

        if not multiple_soundings: plt.show()
        else: return ax1, ax2, ax3




class cpt_plotter( base_method_plotter ):
    def __init__( self, data ):
        super().__init__( data )

        # figure element sizes and positions
        self.sizes_plot = self.generate_sizes_dict( ['q_c','f_s','u_2',], 3 ) #

        # calculate subplot x-positions
        self.sizes_plot['p0'] = self.sizes_plot['margin'] * 0.5 + .012 # method symbol
        self.sizes_plot['p1'] = self.sizes_plot['margin'] # qc
        self.sizes_plot['p2'] = self.sizes_plot['p1'] + self.sizes_plot['q_c'] + self.sizes_plot['margin'] # fs
        self.sizes_plot['p3'] = self.sizes_plot['p2'] + self.sizes_plot['f_s'] + self.sizes_plot['margin'] # u2

        # define total size
        k = 5
        self.fig_width = 8 * ( 1 + 0 )
        self.fig_height = ( 1 + (max(self.data[0][-1],5) // k) * k ) # new increment each 5m


    def draw_method_symbol( self, ax ):
        ax.axis('off')

        base_ang = 27*np.pi/180

        thetas = np.array([base_ang,np.pi-base_ang,3*np.pi/2,base_ang])
        r = 2
        x = r * np.cos( thetas )
        y = r * np.sin( thetas )
        ax.plot( x, y, c=self.colors['method_symbol'] )
        
        ax.axis('equal')
        ax.set_ylim (-r, 2*r )


    def plot( self, multiple_soundings=False ):
        fig = plt.figure( figsize=(self.fig_width, self.fig_height) )

        # show borhole name if given
        if not multiple_soundings:  
            fig.suptitle( self.reference.reference.pos_name, x=self.sizes_plot['p2']/2, fontsize=16)

        n = self.fig_width/self.fig_height/0.75

        # define axis for each figure component
        axm = fig.add_axes( [self.sizes_plot['p0'], 0.96, self.sizes_plot['margin']*0.8, 0.04*n] ) # method figure
        ax1 = fig.add_axes( [self.sizes_plot['p1'], 0.1, self.sizes_plot['q_c'], 0.85] ) # tip resistance ( 0 - XX)
        ax2 = fig.add_axes( [self.sizes_plot['p2'], 0.1, self.sizes_plot['f_s'], 0.85] ) # push force ( 0 - 10)
        ax3 = fig.add_axes( [self.sizes_plot['p3'], 0.1, self.sizes_plot['u_2'], 0.85] ) # porepressure
        ax1.get_shared_y_axes().join(ax1, ax2)
        ax1.get_shared_y_axes().join(ax1, ax3)

        # draw figure components
        self.draw_method_symbol( axm )

        self.draw_curve( ax1, x_vals=self.data[1], y_vals=self.data[0], x_min=0, \
                            color=self.colors['q_c'], suppress_y_axis=False )#, use_ticks=[ 0, 5 ] )
        self.draw_curve( ax2, x_vals=self.data[3], y_vals=self.data[0], x_min=0, \
                            color=self.colors['f_s'], suppress_y_axis=False )#, use_ticks=[ 0, 5 ] )
        self.draw_curve( ax3, x_vals=self.data[2], y_vals=self.data[0], x_min=0, \
                            color=self.colors['u_2'], suppress_y_axis=False )#, use_ticks=[ 0, 5 ] )
        if hasattr( self.reference, 'lowess_data'):
            for lowess_interval in self.reference.lowess_data:
                ax1.plot( self.reference.lowess_data[lowess_interval], self.data[0] )

        ax1.set_ylabel( ylabel='Depth (m)',labelpad=0 )
        self.multicolor_label( ax1,('Tip resistance (kN)',' '),(self.colors['q_c'],self.colors['q_c']),\
                              axis='x',size=10,weight='normal',rotation=0, bbox_to_anchor=(.65, -0.09) )

        if not multiple_soundings: plt.show()
        else: return ax1, ax2, ax3