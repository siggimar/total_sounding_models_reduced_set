import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.interpolate import RBFInterpolator

# script generates figures showing CV validation results.
# as the model is validated at discrete points, scipys rbf
# interpolation function is used to estimate values over
# the full grid.


f_names = {
    0:'0_2m',
    1:'0_3m',
    2:'0_4m',
    3:'0_5m',
    4:'0_8m',
    5:'1_0m',
    6:'1_2m',
    7:'1_4m',
    8:'1_8m',
    9:'2_0m',
}


class knn_plotter():
    def __init__( self ) -> None:
        self.colors = {
            'contours': ( 0, 0, 0 ),
        }
        self.lw = {
            'contours': 1,
        }
        self.ls = {
            'contours': '--',
        }


    def comparison_plot( self, trainer, d_set_a, d_set_b ):
        data = []
        data.append(trainer.get_results(d_set_a))
        #data.append(trainer.get_results(d_set_b))

        if False:
            self.zlims = [min(min(data[0][2]),min(data[1][2])), max(max(data[0][2]),max(data[1][2]))]
            self.zlims[0]=int(self.zlims[0]*50)/50
            self.zlims[1]=int(self.zlims[1]*50)/50 + 0.02
        else: self.zlims=(50,66) #self.zlims=(50,66) #

        fig, ax = plt.subplots( 1,1, figsize=(6,2.5))

        man=None
        if d_set_a==10:
            y1=55
            man=[[(13.8,y1),(26.2,y1),(79.4,y1),(150,y1)],[(9.3,y1),(20,y1)]]
        if d_set_a==15:
            y1 = 75
            y2 = 40
            y3 = 60
            man=[[(13.8,y1),(28.5,y1),(49.5,y1)],[(10.5,y1),(31.5,y1)]] 
            #y1=.85
            man=[[(4.5,y1),(10,y1),(73.4,y1)],[(6.4,y2),(20,y2)]]  # normalized
            man=[[(10.5,y3),(34,y3),(44,52),(52,45),(62.3,40),(78,36),(106,27)],[(17.1,y1),(48.3,y3), (77,35)]] # psi
        man=None


        for i, (ws, ks, AUCs) in enumerate( data ):
            self.draw_ax(fig, ax, ks, ws, AUCs, i, man=man)
        
        #axs[0].set_title( 'GSA-dataset', fontsize=14 )
        #axs[1].set_title( 'Sensitivity-dataset', fontsize=14 )
        ax.set_xlabel( 'Number of neighbors, ' + r'$k$' + ' (-)', fontsize=14 )
        plt.tight_layout()
        plt.savefig('AUC_tot_' + trainer.var + '_' + str(d_set_a) + '_' + str(d_set_b) +'.png', dpi=600)
        plt.show()


    def fmt( self, x ):
        s = f'{x:.2f}'
        #if s.endswith('0'): s = f'{x:.1f}'
        return rf'{s}' if plt.rcParams['text.usetex'] else f'{s}'



    def draw_ax( self, fig, ax, x, y, z, i, man ):
        if False:
            xs, ys, zs = [], [], []
            for some_x, some_y, some_z in zip(x,y,z):
                #if any(np.abs(np.logspace(np.log10(1), np.log10(221), 100) - some_x) < 1):
                if any(np.abs(np.arange(1, 221, 10) - some_x) < 1) and some_y<98:
                    xs.append(some_x)
                    ys.append(some_y)
                    zs.append(some_z)
            x, y, z = xs, ys, zs
        
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list( "", ['blue', 'yellow', 'red'] ) #['red','orange','yellow','green','blue']
        norm = matplotlib.colors.Normalize(vmin=self.zlims[0], vmax=self.zlims[1])

        n_int = 500

        y = list(np.array(y)*100)
        z = list(np.array(z) *100)

        xi, yi = np.logspace(np.log10(min(x)), np.log10(max(x)), n_int), np.linspace(min(y), max(y), n_int)        
        xi, yi = np.meshgrid(xi, yi)

        kernel = [ 'multiquadric', 'inverse', 'gaussian', 'linear', 'cubic', 'quintic', 'thin_plate' ][3]
        rbf = scipy.interpolate.Rbf( x, y, z, function=kernel, degree=3 )
        zi = rbf(xi, yi)
        cen_map = ax.pcolormesh( xi, yi, zi, cmap=cmap, norm=norm )
        contour_ints = np.arange(self.zlims[0], self.zlims[1]+1e-5, 4)        
        c_plt = ax.contour( xi, yi, zi, contour_ints, linestyles=[self.ls['contours']], linewidths=[self.lw['contours']], colors=[self.colors['contours']] )
    
        if man==None: ax.clabel( c_plt, c_plt.levels, inline=True, fmt=self.fmt, fontsize=12 )
        else: ax.clabel( c_plt, inline=True, fmt=self.fmt, fontsize=12, manual=man[i], rightside_up=True )

        if False: 

            fig.subplots_adjust(right=0.8)
            
            cbar_ax = fig.add_axes([0.83, 0.105, 0.03, 0.775])
            cb = fig.colorbar(cen_map, cax=cbar_ax)

            #cb = plt.colorbar( cen_map ) # draw legend            
            cb.set_label(r'$AUC_{total}$' + ' (-)', size=14 )
            cb.ax.set_yticks(contour_ints )  # vertically oriented colorbar
            cb.ax.tick_params( labelsize=12 )
            
        else:
            cb = plt.colorbar( cen_map ) # draw legend 
            c_label=r'$AUC_{total}$' + ' (%)'
            cb.set_label(c_label, size=14 )
            cb.ax.tick_params( labelsize=12 )

        for some_interval in contour_ints: # draw contours on legend
                cb.ax.plot( [0,1], [some_interval]*2, ls=self.ls['contours'], lw=self.lw['contours'], c=self.colors['contours'] )
            
        
        if i==0:
            y_label = 'Window and endpoint, ' + r'$w$' + ' (% length)'
            y_label = r'$w$' + ' and ' + r'$r$' + ' (% length)'
            #ax.set_xticklabels([])
            #y_label = 'Window, ' + r'$w$' + ' (% length)'

        
        ax.set_xticks(np.arange(0,101,10))
        ax.grid( color=(0,0,0) )
        
        #t_a = ax.text(95, 92, chr(ord('A')+i), fontsize=20, color=(0,0,0), verticalalignment='top', horizontalalignment='right', zorder=20)
        #t_a.set_bbox(dict(facecolor=(1,1,1), edgecolor=(1,1,1)))

        y_label = r'$w$' + ' (% length)'
        y_label = r'$w$' + ', ' + r'$r$' + ' (% length)'
        ax.set_ylabel( y_label, fontsize=14 )
        
        
        
        x_ticks = np.arange(20,201,20)
        x_ticks[0]=17
        ax.set_xticks( x_ticks )
        ax.set_yticks( np.arange(0,101,20) )
        
        ax.set_xlim([1,200])
        ax.set_ylim([0,100])
        
        ax.tick_params( axis='both', labelsize=12 )        