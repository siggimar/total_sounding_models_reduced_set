from data_loader import data
import pickle
import os
import numpy as np
from dtaidistance import dtw
import json
import time
import matplotlib.pyplot as plt
from GSA_diagram import GSA_dia


# script is integral to this study.  
# each group is analized using DTW distances
#   1. the center most curve is assigned group 201
#   2. then each additional curve that together with all the rest in group 201 minimizes the least sum DTW distance to all others in the full set is assigned group 201
#   3. process is stopped at N curves (here 20)


data_folder = 'training_data'
accepted = [ 'Leire','Siltig Leire','Leirig Silt', 'Silt', 'Sandig Silt', 'Siltig Sand', 'Sand', 'Grusig Sand', 'Sandig Grus', 'Grus' ]

save_file = 'reduced_sets.pkl'
save_folder = 'saves'
save_path = os.path.join( save_folder, save_file )

def get_data():
    res = {}
    all_data = data( 'training_data', 15, read_json=True )

    for d in all_data.data:
        some_class = all_data.data[d]['labels']['max two class']
        if some_class in accepted:
            some_curve = {
                    'd':np.array(all_data.data[d]['data']['d']),
                    'f_dt':np.array(all_data.data[d]['data']['f_dt']),
                    'q_n':np.array(all_data.data[d]['data']['q_n']),
                    'id': d
                }
            if some_class in res:
                res[ some_class ].append( some_curve )
            else:
                res[ some_class ] = [ some_curve ]
    return res


def calc_reduced():
    max_it = np.inf
    res = {}
    param = [ 'd', 'f_dt', 'q_n' ][2]
    all_data = get_data()
    for soil_class in accepted:
        print( 'working on ' + soil_class )
        S_A = all_data[ soil_class ]
        S_B = [] # points in B
        D_B = []

        n = len( S_A )

        start = time.time()
        dists = np.zeros( shape=(n,n) )
        window = int(0.2 * len(S_A[0][param]))
        for i in range(n):
            for j in range(i+1,n):
                dists[i][j] = dtw.distance_fast( S_A[i][param], S_A[j][param], window=window )
                dists[j][i] = dists[i][j] # DTW(a,b)==DTW(b,a)

        dists_B = np.ones( shape=(n) ) * np.inf

        max_its=min( max_it, n )

        for k in range(max_its):
            it_res = {}
            p = n - len( S_B ) # normalizing factor
            for i in range( n ):
                if i in S_B: continue # ignore B elements
                tmp_B_dists = dists_B.copy()
                tmp_B_dists = np.where( dists[i]<tmp_B_dists, dists[i], tmp_B_dists )
                it_res[ np.sum(tmp_B_dists)/p ] = [ i, tmp_B_dists, S_A[i]['id'] ] # distance as keys

            best_dist, info = min(it_res.items())
            if False:
                best_dist = np.random.choice(list(it_res.keys()))
                info = it_res[best_dist]

            S_B.append( info[2] )
            D_B.append( best_dist )
            dists_B = info[1]
        res[soil_class] = [ S_B, D_B ]

    return res


def res_to_figure( res ):
    # plots figure 5A in NGM paper, with d_si/ds1 on y azis, and steps on the x-axis

    gsadia = GSA_dia()
    fig, ax = plt.subplots( figsize=(6,3.2), tight_layout=True )
    ylims = [np.inf,-np.inf]
    for i, r in enumerate(res):

        y = np.divide( np.array( res[r][1] ), res[r][1][0] ) # np.array( res[r][1] ).copy()
        y = y[:100] * 100
        #y = np.array( res[r][1] ).copy()
        ylims[0] = min(min(y), ylims[0])
        ylims[1] = max(max(y), ylims[0])
        x = np.arange( len(y) ) + 1
        #x = np.divide(x, x[-1]) * 100
        ls = '-' if i%2==0 else '--'
        ax.plot( x, y, c=gsadia.f_colors[r], ls=ls, lw=3, label=gsadia.translate[r] )

    for t, c in zip([20],[(1,0.5,0.2),(.95,.2,.2)]):
        ax.plot([t]*2,[0,100], lw=3, ls='--', zorder=10, c=c, label=r'$t=$' + str(t) )
    ax.set_ylim(ylims)
    ax.set_ylim(0,100)
    ax.set_xlim(0,100)
    ax.set_yticks(np.arange(0,101,10))
    ax.set_xticks(np.arange(0,101,10))
    ax.grid()
    ax.tick_params(axis='both', labelsize=12)
    ax.legend( fancybox=False, framealpha=1, fontsize=11, loc='best', ncols=2 ).set_zorder(200)
    ax.set_ylabel( r'$d_{S.i}$ / $d_{S.1}$ (%)', fontsize=14)
    ax.set_xlabel( 'Iteration, ' + r'$i$ (-)', fontsize=14)
    plt.show()


def save_reduced_sets( id_lists ):
    folder_reduced = 'reduced_set'
    if not os.path.isdir( folder_reduced ): os.mkdir( folder_reduced )

    with open( os.path.join('training_data', 'd_set_rko_1.0m.json'), 'rb' ) as f:
        all_data = json.load( f ) # load all data from JSON

    f_name_template = 'reduced_set_'

    for id_l in id_lists: # leverage groups to utilize previous work
        for k in all_data:
            if k in id_lists[id_l]:
                all_data[k]['coordinate group'] = 201
            else:
                all_data[k]['coordinate group'] = 999
        
        f_path = os.path.join( folder_reduced,f_name_template + str(id_l) + '.json' )
        with open( f_path, 'w') as f:
            f.write( json.dumps( all_data, indent=4 ) )


def reduce_set( plot=False ):
    res = load_res()
    if res is None:
        res = calc_reduced() # calculate reduction
        save_res( res )

    if plot: res_to_figure( res ) # visualize distances during reduction

    id_lists = {} # extract id lists - 
    for n_ids in [20,70]: # from sorted list, select n curves to define reduced set.
        id_list = []
        for s_type in res:
            k = min( len(res[s_type][0]) - 1, n_ids ) # leave one in test set if len(ids)<n_ids
            id_list += res[s_type][0][:k]
            a=1
        id_lists[n_ids] = id_list

    save_reduced_sets( id_lists ) # generate & save sets from ids
    # all done


def load_res():
    if os.path.isfile( save_path ):
        with open( save_path, 'rb' ) as f:
            res = pickle.load( f )
        return res


def save_res( res ):
    with open( save_path, 'wb' ) as f:
        pickle.dump( res, f )


if __name__=='__main__':
    reduce_set( plot=True )