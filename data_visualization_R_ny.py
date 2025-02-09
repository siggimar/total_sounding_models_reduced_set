import data_loader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
import matplotlib.gridspec as gridspec

'''
script to visualize point placements and distribuitions for different soil materials.
used to generate 2D figures to help define a 2D SBT chart

'''

def get_data( n=1, var_1='q_n', var_2='f_dt' ):
    # start with arrays
    x, y, labels = [], [], []

    # import columns from different datasets (0.3m selected)
    X_sens_qn, y_sens_qn, g_sens_qn, all_types_sens_qn = data_loader.load( n, data_column=var_1, min_leaf=1 )
    X_gsa_qn, y_gsa_qn, g_gsa_qn, all_types_gsa_qn = data_loader.load( n+10, data_column=var_1, min_leaf=1 )

    X_sens_fdt, y_sens_fdt, g_sens_fdt, all_types_sens_fdt = data_loader.load( n, data_column=var_2, min_leaf=1 ) # Sensitive-dataset
    X_gsa_fdt, y_gsa_fdt, g_gsa_fdt, all_types_gsa_fdt = data_loader.load( n+10, data_column=var_2, min_leaf=1 ) # GSA-dataset

    # construct label arrays
    Y_sens = [ all_types_sens_qn[yi] for yi in y_sens_qn ] # sensitive
    Y_gsa = [ all_types_gsa_qn[yi] for yi in y_gsa_qn ] # GSA


    for fdt, qn, label in zip(X_sens_fdt, X_sens_qn, Y_sens): # y_axis
        if np.std(fdt)<=0 or np.average(qn)<=0: continue # does not plot on log scale
        if label=='Quick clay': label='Sensitive' # NVE guidelines have 1 class
        if label=='Brittle': label='Sensitive'
        if len(fdt)==0:continue
        y.append( np.sqrt(np.sum(np.power(fdt,2))/len(fdt)) ) # lowess_20 residual passed as fdt var
        x.append( np.average(qn) )
        labels.append( label )

    for fdt, qn, label in zip(X_gsa_fdt, X_gsa_qn, Y_gsa ): # y_axis        
        if np.std(fdt)<=0 or np.average(qn)<=0: continue # does not plot on log scale
        if label=='Quick clay': label='Sensitive'
        if label=='Brittle': label='Sensitive'
        if len(fdt)==0:continue
        y.append( np.sqrt(np.sum(np.power(fdt,2))/len(fdt)) )
        x.append( np.average(qn) )
        labels.append( label )

    return x, y, labels


def data_to_df( x, y, labels, simplify=[] ):
    if simplify:
        # flatten nested lists
        simplify = [it if isinstance(it, int) else it for subl in simplify for it in (subl if isinstance(subl, list) else [subl])]

    # filter data by label
    x = [ xi for xi, li in zip(x, labels) if li in simplify ]
    y = [ yi for yi, li in zip(y, labels) if li in simplify ]
    labels = [ li for li in labels if li in simplify ]

    return pd.DataFrame( {'std(f_dt)': y, 'q_ns': x, 'labels':labels } )


def log_tick_formatter( val, pos=None ):
    return f"$10^{{{val:g}}}$"


def data_to_kde( x, y, logx=True, logy=True, N=100 ):
    ''' function to calculate data meshgrid '''
    n_pts = N #400 goood
    x_ = x if not logx else np.log10( x )
    y_ = y if not logy else np.log10( y )

    if np.any( x<0 ) or np.any( y<0):
        a=1

    xlims = (np.log10(1e-1),np.log10(1e4 ))
    ylims = (np.log10(1e-3),np.log10(1e2 ))

    xmin, xmax, ymin, ymax = x_.min(), x_.max(), y_.min(), y_.max()
    xmin, xmax, ymin, ymax = xlims[0], xlims[1], ylims[0], ylims[1]

    X, Y = np.mgrid[xmin:xmax:(n_pts*1j), ymin:ymax:(n_pts*1j)]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack( [x_, y_] )

    kernel = stats.gaussian_kde( values ) #, 'silverman' )
    Z = np.reshape(kernel(positions).T, X.shape)

    max_index = np.unravel_index(np.argmax(Z), Z.shape)

    xz = Z[:, max_index[1]] # section along x through max_index
    xx = X[:, max_index[1]] # corresponding xs
    yz = Z[max_index[0], :] # section along y ...
    yy = Y[max_index[0], :] # the ys

    x_top, y_top = X[max_index[0]][0], Y[0][max_index[1]]

    #print(max(xz), max(yz)) # 1.029710887258691

    return X, Y, Z, x_, y_, xz, xx, yz, yy, x_top, y_top


def log_tick_formatter( val, pos=None ):
    return f'$10^{{{val:g}}}$'


def set_axis_log_formatter( axis ):
    axis.set_major_formatter( mticker.FuncFormatter(log_tick_formatter) )
    axis.set_major_locator( mticker.MaxNLocator(integer=True) )


def format_jointplot_ax( ax_main, ax_top, ax_side, xlims, ylims, logx, logy, rc, rc_lims, detailed ):
    f_size_axlabels = 14

    xlims_= ( np.log10(l) for l in xlims )if logx else xlims
    ylims_= ( np.log10(l) for l in ylims )if logy else ylims

    if logx:
        set_axis_log_formatter( ax_main.xaxis )
        set_axis_log_formatter( ax_top.xaxis )
    if logy:
        set_axis_log_formatter( ax_main.yaxis )
        set_axis_log_formatter( ax_side.yaxis )

    ax_main.set_xlim( xlims_ )
    ax_main.set_ylim( ylims_ )
    
    if detailed:
        ax_top.set_ylim( ( 0, 1.05 ) )
        ax_side.set_xlim( ( 0, 1.05 ) )

    # format axis
    ax_top.tick_params( axis='both', which='both', bottom=False, top=False, left=False, right=False )
    ax_side.tick_params( axis='both', which='both', bottom=False, top=False, left=False, right=False )
    plt.setp( ax_top.get_xticklabels(), visible=False )
    plt.setp( ax_top.get_yticklabels(), visible=False )
    plt.setp( ax_side.get_xticklabels(), visible=False )
    plt.setp( ax_side.get_yticklabels(), visible=False )

    # turn off spines
    ax_main.spines[ 'top' ].set_visible( False )
    ax_main.spines[ 'right' ].set_visible( False )
    ax_top.spines[ 'top' ].set_visible( False )
    ax_top.spines[ 'left' ].set_visible( False )
    ax_top.spines[ 'right' ].set_visible( False )
    ax_side.spines[ 'top' ].set_visible( False )
    ax_side.spines[ 'right' ].set_visible( False )
    ax_side.spines[ 'bottom' ].set_visible( False )

    # turn off x/y ticklabels inside grid
    if rc[0]!=rc_lims[0]: plt.setp( ax_main.get_xticklabels(), visible=False )
    if rc[1]!=0: plt.setp( ax_main.get_yticklabels(), visible=False )

    if rc[0]==rc_lims[0]: ax_main.set_xlabel( '' + r'$q_{ns}$' + ' (-)', fontsize=f_size_axlabels ) # #r'$F_{DT}$'
    if rc[1]==0: ax_main.set_ylabel( r'$R$' + ' (kN)', fontsize=f_size_axlabels )


def calc_perc_elev( X, Y, Z, perc ):
    ''' 
        calculates the 
        tried Newton-Rapson method but due to discretization the area curve is not smooth.
        using halving method to find precise index
    '''
    
    perc /= 100 # decimals
    eps = 1e-5

    max_iterations = 50 # 13-14 enough for 100x100 gridspec

    def f_z( i, Z_, z_list, dx, dy ):
        return np.sum(np.where(Z_ < z_list[i]+eps, 0, Z_))*dx*dy # A_

    # Integral estimates by riemann sums    
    dx = X[1][0]-X[0][0]
    dy = Y[0][1]-Y[0][0]
    
    # normalize A to 1
    A = np.sum(Z) * dx * dy
    Z_ = Z / A # checks below simpler if area = 1:  verify with: print(np.sum(Z_)*dx*dy)
    
    # available z_values
    z_list = list( set(np.ravel(Z_)) )
    z_list.sort()

    
    i_low = len( z_list ) - 1
    i_high = 0

    visited = {} # visited index register

    visited[i_low] = f_z( i_low, Z_, z_list, dx, dy )
    visited[i_high] = f_z( i_high, Z_, z_list, dx, dy )

    if visited[i_low]>=perc: return z_list[i_low]
    if visited[i_high]<=perc: return z_list[i_high]
    
    for iteration in range( max_iterations ):
        i_test = int( (i_low+i_high)/2 )
        
        if i_test in visited: break # we're there -> return closest match
  
        visited[i_test] = f_z( i_test, Z_, z_list, dx, dy )
        if visited[i_test]==perc: break # exact match

        if visited[i_test]<=perc: i_low=i_test
        elif visited[i_test]>=perc: i_high=i_test

    res_key, res_val = min(visited.items(), key=lambda x: abs(perc - x[1]))
    return z_list[res_key] * A, Z_

def calc_arr_val_z( z_val, x, y ):

    def slices( x, y, val, eps ):
        ind = np.where( np.abs(y - val)<eps )[0]
        x_, y_ = x[ind], y[ind]
        return x_, y_
    
    i_div = np.argmax(y)
    x_1, y_1 = x[0:i_div], y[0:i_div]
    x_2, y_2 = x[i_div:], y[i_div:]

    x_1_, y_1_ = slices( x_1, y_1, z_val, (x_1[-1]-x_1[0])/8 )
    x_2_, y_2_ = slices( x_2, y_2, z_val, (x_1[-1]-x_1[0])/8 )
    x1 = np.interp(z_val, y_1_, x_1_ )
    x2 = np.interp(z_val, np.flip(y_2_), np.flip(x_2_) )
    
    return x1, x2


def jointplot_axs( 
        x=None, y=None, label=None,
        c='k', m='o', i=0,
        ax_main=None, ax_top=None, ax_side=None,
        xlims=(1e-1,1e4), ylims=(1e-3,1e2), logx=True, logy=True, # Haugen et al.'16
        rc=(0,0), rc_lims=(0,0),
        contours=True,
        fill=True,
        fill_dist=True,
        data=True,
        dist=True,
        mode=True,
        perc=50,
        all_data=None,
        N=100,
        detailed=False
        ):
    ''' module to define jointplots,  takes 3 axes and draws data and distributions'''
    fill_alpha = 0.2

    # format all axes
    format_jointplot_ax( ax_main, ax_top, ax_side, xlims, ylims, logx, logy, rc, rc_lims, detailed )

    # get data to plot
    labelm = None # for F plot
    if data:
        all_data = list(data_to_kde( x,y, logx=logx, logy=logy, N=N ))

    if all_data is not None:
        X, Y, Z, x_, y_, xz, xx, yz, yy, x_top, y_top = all_data

    if data:
        labelm = label + ' (' + str(len(x_)) +')'

    # scatter of data + point for mode
    if mode: ax_main.plot( x_top, y_top, marker='o', mec=(0,0,0), mew=1, mfc=c, ms=7, ls='none', label=labelm, zorder=i+20, alpha=1 )
    if data: ax_main.plot( x_, y_, marker=m, mec=(1,1,1), mew=0.6, mfc=c, ms=5, ls='none', zorder=i, alpha=0.5 )

    # percent vol shades
    if data or fill or contours:
        Z_max= yz.max()
        z_perc, Z_ = calc_perc_elev(X, Y, Z, perc )
        all_data[2] = Z_

        if contours:
            linestyles = 'solid' if False else 'dashed' # perc==50
            cont_col = 'k'
            if labelm==None:
                cont_col=c
            CS = ax_main.contour( X, Y, Z, [z_perc,Z_max], linewidths=1.5, colors=[cont_col], linestyles=linestyles, zorder=i+11, alpha=1 )
            


        if fill: # countour plot - optional fill
            CS = ax_main.contourf( X, Y, Z, [z_perc,Z_max], colors=[c], zorder=i+10, alpha=fill_alpha)
    
    if dist: # plot distributions for presented data
        ax_side.plot( yz, yy, c=c, lw=1.5 )
        ax_top.plot( xx, xz, c=c, lw=1.5 )

    if mode:
        ax_side.plot( yz.max(), y_top, marker='o', mec=(0,0,0), mew=1, mfc=c, ms=7, ls='none', zorder=i+20, alpha=1,clip_on=False )
        ax_top.plot( x_top, xz.max(), marker='o', mec=(0,0,0), mew=1, mfc=c, ms=7, ls='none', zorder=i+20, alpha=1,clip_on=False )


    if fill_dist: # shade under distributions
        x_zs = calc_arr_val_z( z_val=z_perc, x=xx, y=xz )
        y_zs = calc_arr_val_z( z_val=z_perc, x=yy, y=yz )
        ax_top.fill_between( x=xx, y1=xz, where=(x_zs[0] < xx)&(xx < x_zs[1]), color=c, alpha=fill_alpha )
        ax_side.fill_betweenx( y=yy, x1=yz, where=(y_zs[0] < yy)&(yy < y_zs[1]), color=c, alpha=fill_alpha )
    
    if not hasattr(ax_main, 'axis_id_txt'):
        ax_main.text(0.025,0.96, chr(ord('A')+i), verticalalignment='top', horizontalalignment='left', transform=ax_main.transAxes, fontsize=16)
        ax_main.axis_id_txt = 0
    else:
        ax_main.axis_id_txt += 1

    return all_data

'''
def get_ax_min_max( ax ):
    x_max = float('-inf')
    x_min = float('inf')
    y_max = float('-inf')
    y_min = float('inf')

    for line in ax.lines:
        x_data = line.get_xdata()
        y_data = line.get_ydata()
        if x_data.max() > x_max:
            x_max = x_data.max()
        if x_data.min() < x_min:
            x_min = x_data.min()
        if y_data.max() > y_max:
            y_max = y_data.max()
        if y_data.min() < y_min:
            y_min = y_data.min()
    return x_min, x_max, y_min, y_max
'''
def summary( n=1, f_name='', detailed=False, N=100 ):
    # available_classes = ['Quick clay', 'Brittle', 'Not sensitive', 'Clay', 'Silty clay', 'Clayey silt', 'Silt', 'Sandy silt', 'Silty sand', 'Sand', 'Gravelly sand', 'Sandy gravel', 'Gravel']

    colors = { # (r,g,b)
        'Sensitive': (237,28,46),
        'Quick clay': (237,28,46),
        'Brittle': (237,28,46), #(220,170,0)
        'Not sensitive': (90,180,50),
        'Clay': (0,160,190),
        'Silty clay': (70,140,130), 
        'Silt': (112,48,160), # dark purple # (80,40,120)
        'Sandy silt': (143, 207, 95),
        'Clayey silt': (70,190,170),
        'Sand': (10,150,100),
        'Gravelly sand': (245, 105, 155),
        'Sandy gravel': (180,180,180),
        'Gravel': (60,60,60),
        'model_gen': (0,0,0,255),
    }
    colors = {
        'Clay': (0, 160, 190),
        'Silty clay': (76, 188, 209),
        'Clayey silt': (154, 110, 188),
        'Silt': (112, 46, 160),
        'Sandy silt': (70, 0, 132),
        'Silty sand': (83, 181, 146),
        'Sand': (10, 150, 100),
        'Gravelly sand': (0, 119, 54),
        'Sandy gravel': (118, 118, 118),
        'Gravel': (60, 60, 60),
        'Sensitive clay': (242, 96, 108),
        'Sensitive': (242, 96, 108),
        'Quick clay': (242, 96, 108),
        'Sensitive silt': (242, 96, 108),
        'Brittle': (251, 181, 56),
        'Not sensitive': (90, 180, 50),
    }

    colors = { k: ( v[0]/255, v[1]/255, v[2]/255, 1 ) for k,v in colors.items() } # -> [0-1] * 4

    markers = { k: 'o' for k,v in colors.items() }
    #[ markers.update( {s:'^'} ) for s in ['Quick clay', 'Silt', 'Gravel'] ]

    desired_labels = [ 'Clay', 'Silt', 'Sand',['Not sensitive', 'Sensitive', 'Gravel'],'model_gen_1', 'model_gen_2' ] # model_gen last!
    if detailed: desired_labels = [['Clay', 'Silty clay'], ['Clayey silt', 'Silt', 'Sandy silt'], ['Silty sand', 'Sand', 'Gravelly sand'], ['Not sensitive', 'Sensitive', 'Sandy gravel', 'Gravel'],'model_gen_2']

    rows, cols = 2, 3

    w_aspect_r = [6,1,0.2] * cols
    h_aspect_r = [1,6,0.2] * rows

    xlims = (1e-1,1e4)
    ylims = (1e-3,1e2)

    vars = ['d','f_dt', 'f_dt_lowess_20', 'f_dt_lowess_30', 'f_dt_lowess_40', 'q_n', 'q_ns', 'f_dt_res']

    x, y, labels = get_data(n=n, var_1=vars[5], var_2=vars[-1]) #   <<<<<<<<<<<<   ############   set i for dataset   ############   >>>>>>>>>>>> Haugen et al: n=1: [5]&[1]
    data = data_to_df( x, y, labels, simplify=desired_labels )

    fig = plt.figure( figsize=(10.5,8))#, tight_layout=True )
    gs = gridspec.GridSpec( rows*3, cols*3, height_ratios=h_aspect_r, width_ratios=w_aspect_r )
    axs = []

    all_x = data['q_ns'].to_numpy()
    all_y = data['std(f_dt)'].to_numpy()
    all_data = data_to_kde(all_x, all_y, logx=True, logy=True, N=N)


    data_registry = {}
    for i, labels in enumerate( desired_labels ):
        r, c = int(i/cols), i%cols

        if axs: # create axis
            axs.append( fig.add_subplot(gs[r*3+1, c*3], sharex=axs[0], sharey=axs[0]) ) # ax_main
        else:
            axs.append( fig.add_subplot(gs[r*3+1, c*3]) ) # ax_main

        axs.append( fig.add_subplot(gs[r*3, c*3], sharex=axs[0]) ) # ax_top            
        axs.append( fig.add_subplot(gs[r*3+1, c*3+1], sharey=(axs[0])) ) # ax_side

        # add data
        if not isinstance( labels, list ): labels = [ labels ] # to_list()
        for label in labels:
            current_data = data.loc[ data['labels']==label ]
            x = current_data['q_ns'].to_numpy()
            y = current_data['std(f_dt)'].to_numpy()
            if len(x)==0:
                print('no data for ' + label)
                continue

            if 'model_gen' not in label:
                data_registry[label] = jointplot_axs(
                    x=x, y=y,
                    label=label,
                    c=colors[label], 
                    m=markers[label],
                    i=i,
                    ax_main=axs[-3], ax_top=axs[-2], ax_side=axs[-1], 
                    xlims=xlims, ylims=ylims, 
                    logx=True, logy=True, 
                    rc=(r,c), rc_lims=(rows-1,cols-1),
                    perc=50,
                    N=N
                    #fill_dist=False,
                    #mode=True ,dist=False,
                )
                if i==0 and label==labels[-1]: axs[-3].plot(xlims, [np.log10(ylims[0]/10)]*2, lw=1.5, ls='--',c='k', label='50% bounded volume' )

        if 'model_gen' in label: # put it together
            selected_model = [['Not sensitive', 'Sensitive'], [ 'Clay', 'Silt', 'Sand', 'Gravel' ]][i-4]
            if detailed: selected_model = [['Clay', 'Silty clay', 'Clayey silt', 'Silt', 'Sandy silt', 'Silty sand', 'Sand', 'Gravelly sand', 'Sandy gravel', 'Gravel'], [ 'Clay', 'Silt', 'Sand', 'Gravel' ]][i-4]
            
            surfaces = []
            for label in selected_model:
                if label not in data_registry: continue
                surfaces.append( data_registry[label][2] )
                X, Y = data_registry[label][0], data_registry[label][1]
                current_data = data.loc[ data['labels']==label ]
                x = current_data['q_ns'].to_numpy()
                y = current_data['std(f_dt)'].to_numpy()
                jointplot_axs(
                    x=x, y=y,
                    label=label,
                    c=colors[label], 
                    m=markers[label],
                    i=i,
                    ax_main=axs[-3], ax_top=axs[-2], ax_side=axs[-1], 
                    rc=(r,c), rc_lims=(rows-1,cols-1),
                    perc=50,
                    contours=False, fill=False, fill_dist=False, data=False, mode=True ,dist=True,
                    all_data=data_registry[label],
                    N=N,
                    detailed=detailed
                )

            # daylight figure
            # indexed colors to use            
            c_ind = [ colors[label] for label in selected_model ]
            c_ind.append((1,1,1,1))

            z_ref = np.max(all_data[3])*0.0025
            # surface stack to max indices
            stacked_surfaces = np.zeros_like( surfaces[0] )
            for surface in surfaces: stacked_surfaces = np.dstack((stacked_surfaces, surface))
            stacked_surfaces = stacked_surfaces[:,:,1:]
            stacked_surfaces = np.dstack((stacked_surfaces, np.zeros_like( surfaces[0] )+z_ref))
            max_indices = np.argmax(stacked_surfaces, axis=-1)


            # remove gravel color if gravel not in labels
            if 'Gravel' not in data_registry and colors['Gravel'] in c_ind:
                c_ind.remove(colors['Gravel'])

            # color array from max indices and indexed colors 
            colored_array = np.ones(surfaces[0].shape + (4,))
            for j in range(len(selected_model)):
                colored_array[max_indices == j] = c_ind[j]
            colored_array = colored_array.reshape(-1, 4)

            # display the image
            axs[-3].pcolormesh( X, Y, np.zeros_like(X), color=colored_array, shading='auto' )

            axs[-2].plot([-10,10], [z_ref]*2, lw=1, c='k', ls='--', zorder=20)
            axs[-1].plot([z_ref]*2, [-10,10], lw=1, c='k', ls='--', zorder=20)
            
            # dummy data for the legend
            for label in selected_model:
                if label not in data_registry: continue
                axs[-3].plot( data_registry[label][9], data_registry[label][10], marker='s', ms=8, mec=colors[label], mfc=colors[label], alpha=1, label=label, zorder=-1, ls='none' )

            if False: # draw intersections between two PDFs ()
                for label_a, label_b in [('Clay', 'Silt')]:
                    X1, Y1, Z1 = data_registry[label_a][0], data_registry[label_a][1], data_registry[label_a][2]
                    X2, Y2, Z2 = data_registry[label_b][0], data_registry[label_b][1], data_registry[label_b][2]
                    Z3  = Z1-Z2
                    Z3  = np.subtract(Z1,Z2)
                    CS = axs[-3].contour( X1,Y1,Z3, [0, 1], linewidths=2, colors=['k'], zorder=99, alpha=1 )

        if detailed:
            if i<4: axs[-3].legend(loc='lower right')
            else: axs[-3].legend(bbox_to_anchor=(1.16, 1.03), loc='upper left')
        else:
            axs[-3].legend(loc='lower right')

    plt.subplots_adjust( wspace=0.00, hspace=0.00 ) # space between figures
    plt.subplots_adjust( left=0.1, right=0.99, bottom=0.1, top=0.99 )    
    if f_name!='':
        plt.savefig( f_name,dpi=600, transparent=False )
        plt.close()
    else: plt.show()
        


if __name__=='__main__':
    
    for i in [1]:#range(1,10):
        summary(n=i, f_name='fig_6_R_simplified_03_SBT_chart.png', detailed=False, N=500) # 'all_classes_' + str(i) + '.png', all_classes_' + str(i) + '.png # N > 300 useable - 500 used in paper.