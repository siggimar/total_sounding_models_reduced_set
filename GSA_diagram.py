import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


# class to generate a pretty grain size analysis diagram

class GSA_dia():
    def __init__( self ):
        self.counts = {}
        self.pos = {'top':0.89, 'bott':0.08, 'left':0.06, 'width':.92, 'height':.82, 'top_h':0.1}
        self.fractions = { 'Clay':[0.001,0.002], 'Silt':[0.002, 0.063], 'Sand':[0.063,2], 'Gravel': [2,63], 'Stone':[63,100] }
        # colors for this diagram as well as labeled data
        self.f_colors = {
            'Clay':[0,160,190], 
            'Silt':[110,45,160], 
            'Sand':[10,150,100], 
            'Gravel': [60,60,60], 
            'Stone':[0,0,0],
            'Leire':[0,160,190], 
            'Siltig Leire':[75,190,210],
            'Leirig Silt':[155,110,190],
            'Silt':[110,45,160], 
            'Sandig Silt':[70,0,130],
            'Siltig Sand':[80,180,145],
            'Sand':[10,150,100], 
            'Grusig Sand':[0,120,55],
            'Sandig Grus':[118,118,118],
            'Grus': [60,60,60],
        }
        self.translate = {'Leire':'Clay','Siltig Leire':'Silty clay','Leirig Silt':'Clayey silt','Silt':'Silt','Sandig Silt':'Sandy silt',
                     'Siltig Sand':'Silty sand','Sand':'Sand','Grusig Sand':'Gravelly sand','Sandig Grus':'Sandy gravel','Grus':'Gravel'}
        self.f_colors = { k:[v[0]/255, v[1]/255,v[2]/255 ] for k,v in self.f_colors.items() }
        self.f_alpha = 0.2
        self.fontsize_axis_title = 18
        self.fontsize_intermediate = 13
        self.fontsize_main = 14
        self.fontsize_ticks = 12
        self.legend_fontsize = 16
        self.border_lw = .5
        self.xlims = [ 1e-3, 100 ]
        self.ylims = [ 0, 100 ]

        self.fig = plt.figure( figsize=(15, 9), tight_layout=True )
        self.axs = []

        self.add_top()
        self.add_main()


    def add_legend( self ):
        legend_handles = []
        
        for l in [ 'Leire','Siltig Leire','Leirig Silt','Silt','Sandig Silt','Siltig Sand','Sand','Grusig Sand','Sandig Grus','Grus' ]:
            count_txt = ' (' + str( self.counts[l] ) + ')'
            legend_handles.append( mpatches.Patch( facecolor=self.f_colors[l], label=self.translate[l] + count_txt, edgecolor=(0,0,0) ) )
        
        self.axs[-1].legend(loc='lower right', handles=legend_handles, fancybox=False, framealpha=1, fontsize=self.legend_fontsize).set_zorder(200)

    def plot_curve( self, x, y, lw=1, c=(0,0,0), ls='-', label=None ):
        self.axs[-1].plot( x, y, lw=lw, c=c, ls=ls, label='', zorder=np.random.randint(10,110) )
        if label is not None: # count curves for chart legend
            if label in self.counts: self.counts[label] += 1
            else: self.counts[label] = 1


    def plot_range( self, range, lw=2, c=(0,0,0), ls='-' ):
        self.axs[-1].plot( range[0], range[1], lw=lw, c=c, ls=ls, zorder=2 ) # low
        self.axs[-1].plot( range[0], range[2], lw=lw, c=c, ls=ls, zorder=2 ) # high

        self.axs[-1].fill_between(range[0], range[1], range[2], color=c, alpha=0.4)

    def add_top( self ):
        top = self.fig.add_axes( [self.pos['left'], self.pos['top'], self.pos['width'], self.pos['top_h']] ) # rate of penetration

        top.set_xlim( self.xlims )
        top.set_ylim( 0, 1 )
        top.set_xscale('log')
        top.minorticks_off()
        top.set_xticks([])
        top.set_yticks([])

        # border around fraction definitions
        top.plot( [self.xlims[0]]+ self.xlims + list(reversed(self.xlims)),[1,0,0,1,1],lw=self.border_lw, c=(0,0,0), clip_on=False )

        for fr in self.fractions:
            lims = self.fractions[fr]
            log_lims = np.log10(lims)
            l_frac = (log_lims[1]-log_lims[0])/6 # txt and intermediate fraction x_vals
            label_x = np.power(10, np.average(log_lims)) # label position

            top.plot( [lims[0]]*2, [0,1], lw=self.border_lw, ls='-', c=(0,0,0) ) # main fraction vertical separation

            y_txt = 0.5 # main fraction y
            if fr.lower() not in ['clay', 'stone' ]:
                y_txt = 0.75
                top.plot( lims, [0.5]*2, lw=self.border_lw, ls='-', c=(0,0,0) ) # horizontal intermediate line

                xs = [ np.power(10, log_lims[0] + l_frac * i)  for i in np.arange(1,6) ]                
                top.plot( [xs[1]]*2, [0,0.5], lw=self.border_lw, ls='-', c=(0,0,0) ) # certical intermediate lines
                top.plot( [xs[3]]*2, [0,0.5], lw=self.border_lw, ls='-', c=(0,0,0) ) # vertical intermediate lines

                for i, txt in enumerate( ['Fine', 'Medium', 'Coarse'] ):
                    top.text( xs[0+2*i], 0.25, txt, size=self.fontsize_intermediate, verticalalignment='center', horizontalalignment='center' ) # vertical intermediate lines
            top.text( label_x, y_txt, fr, size=self.fontsize_main, verticalalignment='center', horizontalalignment='center' ) # main fraction

        top.set_xlim( self.xlims )
        top.set_ylim( 0, 1 )
        self.axs.append( top )


    def add_main( self ):
        main = self.fig.add_axes( [self.pos['left'], self.pos['bott'], self.pos['width'], self.pos['height']] ) # rate of penetration

        # main axis setup
        main.set_ylabel('% Finer', fontsize=self.fontsize_axis_title)
        main.set_xlabel('Grain diameter', fontsize=self.fontsize_axis_title)
        main.set_xlim( self.xlims )
        main.set_ylim( 0, 1 )
        main.set_xscale('log')

        # ticks and ticklabels
        main.set_yticks(np.arange(0,101,20))
        main.set_xticks([0.001,0.002,0.0063,0.01,0.02,0.063,0.1,0.2,0.63,1,2,6.3,10,20,63])
        main.set_xticklabels([1,r'2$\mu$m',6.3,'',20,63,'',200,r'630$\mu$m','','2mm', 6.3,'',20, 63])
        main.tick_params(axis='both', labelsize=self.fontsize_ticks)

        for fr in self.fractions: # fraction colors and support lines
            lims = self.fractions[fr]
            log_lims = np.log10(lims)
            frac = (log_lims[1]-log_lims[0])/3
            
            #rect_c = self.f_colors[fr] + [self.f_alpha]
            #rect = matplotlib.patches.Rectangle((lims[0],0), lims[1]-lims[0], 100, color=rect_c, zorder=-2)
            #main.add_patch(rect) # adds background colors

            main.plot( [lims[0]]*2,[0,100], lw=self.border_lw, ls='-', c=(0,0,0) )

            if fr.lower() not in ['clay', 'stone' ]:
                for i in range(2): # intermediate lines
                    x_val = np.power(10, log_lims+(i+1)*frac )
                    main.plot( [x_val]*2,[0,100], lw=self.border_lw, ls='-', c=(0,0,0,.5) )
        for lvl in main.get_yticks():
            main.plot( self.xlims,[lvl]*2, lw=self.border_lw, ls='-', c=(0,0,0,.5) )

        main.plot( [self.xlims[0]]+ self.xlims + list(reversed(self.xlims)),[100,0,0,100,100],lw=self.border_lw, c=(0,0,0), clip_on=False )
        self.axs.append( main )


if __name__=='__main__':
    gsad = GSA_dia()

    plt.show()