from data_loader import load
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.lines as lines
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, GroupShuffleSplit
from scipy.special import kl_div
import pickle

cmap_cv = plt.cm.coolwarm
cmap_data = plt.cm.viridis#Paired
c_idx = None
color_norm = None

''' Script to compare functionality of CV funtions
    modified from site: Visualizing cross-validation behavior in scikit-learn
'''


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
colors = { k: (v[0]/255,v[1]/255,v[2]/255) for k, v in colors.items() }


def plot_cv_indices( cv, X, y, group, ax, n_splits, ax_id, used_types, lw=10 ):
    """Create a sample plot for indices of a cross-validation object."""

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate( cv.split(X=X, y=y, groups=group) ):        
        
        # Fill in indices with the training/test groups
        indices = np.array([(0,0,0)] * len(X)).astype(object)
        indices[tt] = ( 0, 142/255, 194/255 )
        indices[tr] = ( 242/255, 242/255, 242/255 )

        g_tr =  group[tr]
        g_tt =  group[tt]
        if g_tt[0] in g_tr:
            print('never end up here')
        else:
            pass#print(set(g_tt)) # groups in test set
        
        ax.scatter( # visualizes splits
            range( len(indices) ),
            [ii + 0.5] * len(indices),
            c=indices,
            marker='_',
            lw=lw,
            #cmap=cmap_cv,
            #vmin=-0.2,
            #vmax=1.2,
        )

    lf='group_colors_prv.pkl' if 'prv' in used_types['filename'] else 'group_colors_rko.pkl'

    with open(lf, 'rb') as f: # group colors loaded from dbscan process
        group_colors = pickle.load( f )

    c_sc = [ colors[used_types[some_y]] for some_y in y ] # soil class color

    c_g = [group_colors[g] for g in group]
    ax.scatter( range(len(X)), [ii + 1.5] * len(X), c=c_sc, marker='_', lw=lw )
    ax.scatter( range(len(X)), [ii + 2.5] * len(X), c=c_g, marker='_', lw=lw )

    # Formatting
    y_title = '       Split'
    x_title = 'Sample index'
    yticklabels = list(range(n_splits)) + ['class', 'group']

    info = used_types['filename'].split('_')[-1].rsplit('.',1)[0]

    y_title += ' - ' + r'$\Delta$' + 'D=' + info

    if ax_id[1]!=0:
        #yticklabels = [''] * len(yticklabels)
        #ax.set_yticks([])
        y_title = ''

    if ax_id[0]!=1:
        x_title = ''
        ax.set_xticks([])

    ax.set(
        yticks = np.arange(n_splits + 2) + 0.5,
        yticklabels = yticklabels,
        ylabel = y_title,
        ylim = [n_splits + 2.2, -0.2],
        xlim = [0, len(X)]
    )
    ax.tick_params( axis='both', labelsize=12 )
    ax.set_xlabel( x_title, fontsize=14 )
    ax.set_ylabel( y_title, fontsize=14 )

    if ax_id[0]==0: ax.set_title('{}'.format('\n'+type(cv).__name__), fontsize=12)
    if ax_id[1]!=0: ax.set_yticks([])
    return ax


def cv_function_comparison():
    datasets = [ 3, 5 ] # 2 or more

    k = 10 # desired folds
    j = -1
    fig, ax = plt.subplots( len(datasets), 4, figsize=(12, 3.5*len(datasets)), tight_layout=True )
    if len(datasets)==1: ax=[ax]


    type_color = {}

    for idx in datasets:
        j += 1 
        jj = 2       
        for jdx in range(2):
            jj -= 1
            # load CVs
            gkfold = GroupKFold( n_splits=k ) # GroupShuffleSplit(n_splits=k) # - GSS: uses data from same groups multiple times!
            sgkfold = StratifiedGroupKFold( n_splits=k )

            # load data
            idx_data = idx + 10*jdx
            X, y, g, all_types = load( idx_data )
            X, y, g = sort_data( X, y, g )
            mix_colors( g )


            if len(set(y))<5: # avoid color collision between GSA & Sensitivity data
                delta_y = 20
                y +=delta_y
                all_types = { (k+delta_y if isinstance(k,int) else k):v for k, v in all_types.items() }
            

            for key, value  in all_types.items():
                if not isinstance(key, int): continue
                if value not in type_color:
                    type_color[value] = colors[value]

            plot_cv_indices( gkfold, X, y, g, ax[j][jj*2], k, (j, jj*2), all_types, lw=10  )
            plot_cv_indices( sgkfold, X, y, g, ax[j][jj*2+1], k, (j, jj*2+1), all_types, lw=10 )

    
    c_labels = [ Patch(color=(0, 142/255, 194/255) ), Patch(color=(242/255, 242/255, 242/255) )]
    l_labels = [ 'Test set', 'Train set', ]
    for tck, tcv in type_color.items():
        c_labels.append( Patch(color=tcv) )
        l_labels.append( tck )

    ax[0][-1].legend(
        c_labels,
        l_labels,
        loc=(1.02, 0.0), #loc=(1.02, -0.19),
    )

    # annotate at figure level
    ht=0.96
    fig.text( 0.27, ht, 'GSA-dataset', fontsize=16, color=(0,0,0), verticalalignment='top', horizontalalignment='center' )
    fig.text( 0.68, ht, 'Sensitivity-dataset', fontsize=16, color=(0,0,0), verticalalignment='top', horizontalalignment='center' )

    fig.add_artist( lines.Line2D([.472]*2, [0.01, ht], lw=1, color=(0,0,0)) )
    fig.text( .09, ht, 'A', fontsize=18, color=(0,0,0), verticalalignment='top', horizontalalignment='center' )
    fig.text( .495, ht, 'B', fontsize=18, color=(0,0,0), verticalalignment='top', horizontalalignment='center' )

    fig.subplots_adjust( right=0.7 )

    plt.savefig('CV.png',dpi=600, transparent=False)
    plt.show()


def categorical_cross_entropy( y_true, y_pred ):
    return -np.sum( y_true * np.log(y_pred + 10**-100) )


def score_cv_functions( n ):
    k=10

    cvs = [ GroupKFold, StratifiedGroupKFold]#, GroupShuffleSplit ] # GSS uses data from same groups multiple times (undesired behavior)
    cvs = [ cv( n_splits=k ) for cv in cvs]

    all_res = {}
    for idx in range( n ):        
        # load dataset version
        X, y, g, all_types = load( idx )

        n_classes = len( set(y) )
        population_distr = np.bincount( y, minlength=n_classes ) / np.sum( np.bincount(y, minlength=n_classes) )

        for cv in cvs:
            cv_id = type(cv).__name__
            if not cv_id in all_res: all_res[cv_id]={'KL':{}, 'CCE':{}}
            calc_scores( cv, X, y, g, n_classes, k, population_distr, all_res[cv_id], idx )

    fig, axs = plt.subplots( 1, 2, figsize=(12,5), tight_layout=True )
    its = np.arange( k )

    c1, c2, c3 = 0, 0.2, 0.8
    colors={
        'GroupKFold': (c1,c2,c3),
        'StratifiedGroupKFold':(c3,c2,c1),
        'GroupShuffleSplit': (c1,c3,c2)
    }

    for cv in all_res:
        for measure in all_res[cv]:
            if measure=='CCE': break
            
            for dataset_ver in all_res[cv][measure]:
                if dataset_ver%10==0: vals = np.array([]) # reset average array
                ax_id = 1 - int(dataset_ver/10)
                axs[ ax_id ].plot( its, all_res[cv][measure][dataset_ver], c=colors[cv], ls='--', lw=.8, alpha=.5 )
                vals = np.append(vals, all_res[cv][measure][dataset_ver] )

                if (dataset_ver+1)%10==0: 
                    axs[ ax_id ].plot( its, [np.average(vals)]*len(its), c=colors[cv], ls='-', lw=3, label= 'Average - ' + cv )
                    axs[ ax_id ].plot( its-10, all_res[cv][measure][dataset_ver], c=colors[cv], ls='--', lw=.8, alpha=.5, label=cv + ' - each dataset version' ) # out of view

    for ax in axs:
        ax.set_xticks(its)
        ax.set_xlabel('Split, ' + r'$i$' + ' (-)', fontsize=16)
        ax.set_ylabel(r'$D_{KL}$' + ' (-)', fontsize=16) # = \sum_{j}^{} P(x_j)log(\frac{P(x_j)}{Q(x_j)})
        ax.set_ylim((0,0.2))
        ax.set_xlim((its[0],its[-1]))

    ht = 0.98
    axs[0].set_title( 'GSA-dataset', fontsize=14 )
    axs[1].set_title( 'Sensitivity-dataset', fontsize=14 )
    fig.text( .015, ht, 'A', fontsize=18, color=(0,0,0), verticalalignment='top', horizontalalignment='center' )
    fig.text( .51, ht, 'B', fontsize=18, color=(0,0,0), verticalalignment='top', horizontalalignment='center' )


    for ax in axs:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    handles, labels = plt.gca().get_legend_handles_labels()
    idx = np.argsort(labels).astype(int)

    plt.legend( [handles[i] for i in idx], [labels[i] for i in idx] )
    plt.show()


def calc_scores( CV, X, y, g, n_classes, k, pop_dist, cv_res, idx ):
    cv_res['KL' ][idx]  = []
    cv_res['CCE'][idx] = []

    for ii, (train_index, test_index) in enumerate( CV.split(X=X, y=y, groups=g) ):
        if ii==8 and False:
            cv_res['KL' ][idx].append(  0.025 )
            cv_res['CCE'][idx].append( 0.025 )
            continue
        y_test = y[test_index]
        sample_distr = np.bincount( y_test, minlength=n_classes ) / np.sum( np.bincount(y_test, minlength=n_classes) )
        cv_res['KL' ][idx].append( sum(kl_div(sample_distr, pop_dist)) )
        cv_res['CCE'][idx].append( categorical_cross_entropy(pop_dist, sample_distr) )


def mix_colors( g ):
    # calc point color scheme
    global color_norm
    global c_idx

    l_min, l_max = min(g), max(g)
    l_range = (l_max-l_min)+1
    color_norm = plt.Normalize( l_min, l_max )
    c_idx = np.linspace( l_min, l_max, num=l_range).astype(int)

    np.random.seed(1234)
    np.random.shuffle(c_idx) # shuffles colors around ( better contrast )


def color_by_index( index, colormap_name='gist_rainbow' ):
    colormap = plt.get_cmap(colormap_name)
    return colormap( color_norm(c_idx[index]) ) # randomized index order


def sort_data( X, y, g, sort_classes=True ):
    # first sort by groups,  then optianlly by class within groups
    counts = np.bincount( g )
    sorted_indices = np.argsort(counts)[::-1]

    #print(sorted_indices[:2])
    #print(np.sum(counts[sorted_indices[:2]])/np.sum(counts))

    ii = np.array([]).astype(int)
    for s in sorted_indices:
        ii = np.append(ii, np.where(g==s))

    X, y, g = X[ii], y[ii], g[ii]

    if sort_classes:        
        all_groups = []
        [ all_groups.append(some_g) for some_g in g if some_g not in all_groups ]
        jj = np.array([]).astype(int)
        for some_group in all_groups:
            group_indices = np.where(g==some_group)[0]
            group_vals = y[group_indices]
            jj = np.append( jj, group_indices[np.argsort(group_vals)] )
        X, y, g = X[jj], y[jj], g[jj]

    for kk in jj:
        if kk not in ii:
            a=1

    a=1

    return X, y, g


def test( n ):
    X, y, g, all_types = load( n )
    X, y, g = sort_data( X, y, g )
    mix_colors( g )

    sgkfold = StratifiedGroupKFold( n_splits=10 )
    fig, ax = plt.subplots()
    plot_cv_indices( sgkfold, X, y, g, ax, 10, (1,0), all_types, lw=10  )
    plt.show()



if __name__=='__main__':
    cv_function_comparison()
    score_cv_functions( n=20 )
    #test( n=10 )