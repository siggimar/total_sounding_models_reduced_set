import os
import pickle
from sklearn.metrics import accuracy_score, f1_score
from classifiers import tot_sbt, time_series_KNeighborsClassifier, random_classifier, simple_sens_classifier
from sklearn.model_selection import StratifiedGroupKFold
from data_loader import load
from scipy.interpolate import RBFInterpolator
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

 # script for model validation, used to produce figures for a draft paper.

def acc_calc( dataset, w, k, fuzzy_matching=0 ): # not used!: faster version:  acc_calc_n() below
    f_name = 'saves/acc_scores.pkl'
    res = load_save( f_name )
    
    qn, D, y, g, all_types = load( dataset, data_column='q_n', return_depth=True )
    cv = StratifiedGroupKFold( n_splits=10 )

    for some_w in w:
        for some_k in k:
            print( 'working on ' + str((dataset, some_k, some_w)) )
            if (dataset, fuzzy_matching, some_k, some_w) in res: 
                print('found it saved')
                continue # only calculate once

            dtw_clf = time_series_KNeighborsClassifier( n_neighbors=some_k )
            dtw_clf.fit( qn, y*0 ) # labels removed (supplied for set_w command)
            dtw_clf.set_w( some_w )

            acc_dtw = []
            for ii, (tr, tt) in enumerate( cv.split(X=qn, y=y, groups=g) ): 
                if len(tr)>len(tt): continue # testing accuracy of reduced set
                qn_train, y_train = qn[tr], y[tr]
                qn_test, y_test = qn[tt], y[tt]

                # fit and predict
                dtw_clf.fit( qn_train, y_train )
                y_pred_dtw = dtw_clf.predict( qn_test )
                y_pred_dtw = fuzzy_match( y_test, y_pred_dtw ) # allow fuzzy - all cases

                #score = accuracy_score(y_test, y_pred_dtw)
                #score = f1_score(y_test, y_pred_dtw)
                acc_dtw.append( score )
                

            res[ (dataset, fuzzy_matching, some_k, some_w ) ] = np.average( acc_dtw )
            save_res( f_name, res )


def acc_calc_n( dataset, w, k, fuzzy_matching=0, f_name = 'saves/acc_scores.pkl' ):
    
    res = load_save( f_name )

    print('Training on dataset ' + str(dataset))
    d_f_name = 'reduced_set_' + str(dataset) + '.json'
    d_folder = 'proposed_model'
    f_path = os.path.join( d_folder, d_f_name)
    
    qn, D, y, g, all_types = load( dataset, data_column='q_n', return_depth=True, f_path=f_path )
    
    for i in range(np.random.randint(1000)): np.random.shuffle(g) # removes DBSCAN efforts

    cv = StratifiedGroupKFold( n_splits=10 )

    for some_w in w:
        print( 'working on ' + str((dataset, some_w)) + ' & all k-s provided' )
        if (dataset, fuzzy_matching, k[-1], some_w) in res: 
            print('found it saved')
            continue # only calculate once
        dtw_clf = time_series_KNeighborsClassifier( n_neighbors=k ) # <-- pass np.ndarray with k-s!
        dtw_clf.fit( qn, y*0 ) # labels removed (supplied for set_w command)
        dtw_clf.set_w( some_w )

        accs_dtw = [[] for i in range(len(k))] # accs_dtw = [ [] ] * len(k) # was a source of severe frustration
        for ii, (tr, tt) in enumerate( cv.split(X=qn, y=y, groups=g) ): 
            if len(tr)>len(tt): continue # testing accuracy of reduced set
            print(str(ii+1) + '/10' )
            qn_train, y_train = qn[tr], y[tr]            
            qn_test, y_test = qn[tt], y[tt]

            # fit and predict
            dtw_clf.fit( qn_train, y_train )
            #dtw_clf.fit( qn_test, y_test ) # used to verify 100% train-accuracy: a big no-no in final implementation
            y_preds_dtw = dtw_clf.predict( qn_test )
            
            for i, y_pred_dtw in enumerate( y_preds_dtw ):
                y_pred_dtw = fuzzy_match( y_test, y_pred_dtw, fuzzy_matching ) # allow fuzzy - all cases
                score = accuracy_score(y_test, y_pred_dtw)
                score = f1_score(y_test, y_pred_dtw, average='macro') # micro
                accs_dtw[i].append( score )

        for i, (some_k, acc_dtw) in enumerate(zip(k, accs_dtw)):
            res[ (dataset, fuzzy_matching, some_k, some_w ) ] = np.average( acc_dtw )
        save_res( f_name, res )


def load_save( f_name ):
    if not os.path.isfile( f_name ): return {}
    with open(f_name, 'rb') as f:
        res = pickle.load( f )
    return res


def save_res( f_name, res ):
    with open(f_name, 'wb') as f:
        pickle.dump( res, f )



def optimal_threshold( dataset=5 ):
    ylims = [40,70]
    xlims = [-100,200]
    qn, std_fdt, qns, y, g, all_types = prep_data( dataset=dataset )

    m = np.logical_and(qns>0, std_fdt>0) # remove data with 0/neg registrations of q/std_fdt
    qn, std_fdt, qns, y, g = qn[m], std_fdt[m], qns[m], y[m], g[m]

    # Classes are : 0:quick clay, 1:Brittle, 2:Not sensitive
    # Renamed to    0:Not sensitive, 1:Sensitive

    #print(np.bincount(y)) # [1140 1700 3798] @ dataset 5
    y[y == 0] = 1 # add quick clay to sensitive index (Brittle already there)
    y[y == 2] = 0 # move Not sensitive to index 0
    prop = np.bincount(y) # [3798 2840]


    clf = simple_sens_classifier()
    clf_bound = simple_sens_classifier(apply_bounds=True)

    ts = np.arange(-100,201)
    #ts = np.arange(-100,201,15)
    t, acc, acc_b = [], [], []

    n=len(ts)

    for i, some_t in enumerate(ts):
        if i%10==0:
            print(str(round(i/n*100,1)) + '% done', end='\r')
        clf.set_threshold( some_t )
        clf_bound.set_threshold( some_t )

        y_pred = clf.predict( qns, std_fdt )
        y_pred_b = clf_bound.predict( qns, std_fdt )
        
        t.append(some_t)
        acc.append(accuracy_score(y, y_pred))
        acc_b.append(accuracy_score(y, y_pred_b))

    acc = np.array(acc) * 100
    acc_b = np.array(acc_b) * 100

    idx = np.argmax(acc)    
    print('Optimal threshold: t=' + str(t[idx]) + '. Accuracy: acc=' + str(np.round(acc[idx],2)) + '.' )

    # plot figure
    fig, ax = plt.subplots( figsize=(13,4), tight_layout=True)

    # asymptotes
    ax.plot( [t[0],t[-1]], [prop[0]/np.sum(prop)*100]*2, c=(.3,.3,.3), lw=1.5, ls='--' )
    ax.plot( [t[0],t[-1]], [prop[1]/np.sum(prop)*100]*2, c=(.3,.3,.3), lw=1.5, ls='--' )

    # max value
    ax.plot( [t[idx]]*2, [ylims[0],acc[idx]], c=(.3,.3,.3), lw=1.5, ls='--' )
    ax.plot( [t[idx]], [acc[idx]], marker='o', ms=6, mec=(0,0,0), mfc=(1,1,1), lw=1.5, ls='none', zorder=20 )

    ax.text( t[idx]+3, acc[idx]+0.5, '(' + str(t[idx]) + ', ' + str(round(acc[idx],1)) + '%)', fontsize=14 )

    ax.annotate(' 0 ≤ P ≤ 100', xy=(120,prop[0]/np.sum(prop)*100), xycoords='data', xytext=(105,63), textcoords='data', va='top', ha='left', fontsize=14, arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    ax.annotate('Everything classified sensitive', xy=(110,prop[1]/np.sum(prop)*100), xycoords='data', xytext=(90,50), textcoords='data', va='top', ha='left', fontsize=14, arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    ax.annotate('Nothing classified sensitive', xy=(-50,prop[0]/np.sum(prop)*100), xycoords='data', xytext=(-70,50), textcoords='data', va='top', ha='left', fontsize=14, arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    
    # data
    ax.plot( t, acc_b,c=(237/255,28/255,46/255), lw=2, ls='--' )
    ax.plot( t, acc,c=(0,0,0), lw=2.5 )

    ax.set_xlabel( 'Threshold, ' + r'$t$' + ' (-)', fontsize=14 )
    ax.set_ylabel( 'Accuracy (%)', fontsize=14 )
    ax.tick_params( axis='both', labelsize=12 )
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()


def acc_calc_sens_n( dataset, w, k, fuzzy_matching=0, f_name='saves/acc_scores.pkl' ):
    res = load_save( f_name )

    qn, std_fdt, qns, y, g, all_types = prep_data( dataset=dataset )

    m = np.logical_and(qns>0, std_fdt>0) # remove data with 0/neg registrations of q/std_fdt
    qn, std_fdt, qns, y, g = qn[m], std_fdt[m], qns[m], y[m], g[m]

    # Classes are : 0:quick clay, 1:Brittle, 2:Not sensitive
    # Renamed to    0:Not sensitive, 1:Sensitive

    #print(np.bincount(y)) # [1140 1700 3798] @ dataset 5
    y[y == 0] = 1 # add quick clay to sensitive index (Brittle already there)
    y[y == 2] = 0 # move Not sensitive to index 0

    #print(np.bincount(y)) # [3783 2806]

    cv = StratifiedGroupKFold( n_splits=10 )

    for some_w in w:
        print( 'working on ' + str((dataset, some_w)) + ' & all k-s provided' )
        if (dataset, fuzzy_matching, k[-1], some_w) in res: 
            print('found it saved')
            continue # only calculate once
        dtw_clf = time_series_KNeighborsClassifier( n_neighbors=k ) # <-- pass np.ndarray with k-s!
        dtw_clf.fit( qn, y*0 ) # labels removed (supplied for set_w command)
        dtw_clf.set_w( some_w )

        accs_dtw = [[] for i in range(len(k))] # accs_dtw = [ [] ] * len(k) # was a source of severe frustration
        for ii, (tr, tt) in enumerate( cv.split(X=qn, y=y, groups=g) ): 
            print(str(ii+1) + '/10' )
            qn_train, y_train = qn[tr], y[tr]
            qn_test, y_test = qn[tt], y[tt]

            # fit and predict
            dtw_clf.fit( qn_train, y_train )
            y_preds_dtw = dtw_clf.predict( qn_test )
            
            for i, y_pred_dtw in enumerate( y_preds_dtw ):
                y_pred_dtw = fuzzy_match( y_test, y_pred_dtw, fuzzy_matching ) # allow fuzzy - all cases
                score = accuracy_score(y_test, y_pred_dtw)
                accs_dtw[i].append( score )

        for i, (some_k, acc_dtw) in enumerate(zip(k, accs_dtw)):
            res[ (dataset, fuzzy_matching, some_k, some_w ) ] = np.average( acc_dtw )
        save_res( f_name, res )



def compare_GSA_classifiers( fuzzy_matching=0, dataset=15 ):
    qn, std_fdt, qns, y, g, all_types = prep_data( dataset=dataset )

    if False: # shuffle groups to check effect on accuracy
        np.random.shuffle( g )
        
    
    
    cv = StratifiedGroupKFold( n_splits=10 )
    acc_dtw, acc_sbt, acc_rnd = [], [], []

    sbt_clf = tot_sbt()
    sbt_clf.fit( all_types ) # fits data&clf indexes by labels ( SBT given as 1-10, where data is defined 0-9 )
    dtw_clf = time_series_KNeighborsClassifier(n_neighbors=95) #95
    rnd_clf = random_classifier()
    dtw_clf.fit( qn, y*0 ) # labels removed (supplied for set_w command)
    dtw_clf.set_w( 0.2 ) # 60% of sequence length
    rnd_clf.fit(qn, y) # extracts possible values for y


    for ii, (tr, tt) in enumerate( cv.split(X=qn, y=y, groups=g) ):
        print(str(ii+1) + '/10')
        # create train test splits
        qn_train, y_train = qn[tr], y[tr]
        qn_test, y_test = qn[tt], y[tt]
        std_fdt_test, qns_test = std_fdt[tt], qns[tt]

        # fit knn-dtw and predict for both
        dtw_clf.fit( qn_train, y_train ) # keeps full set stored as classifier.X and classifier.y
        
        y_pred_dtw = dtw_clf.predict( qn_test )
        y_pred_sbt = sbt_clf.predict( qns_test, std_fdt_test ) # asks for x and y (here y is the y-coordinate - not label)
        y_pred_rnd = rnd_clf.predict( qn_test )

        if fuzzy_matching>0:
            y_pred_dtw = fuzzy_match( y_test, y_pred_dtw, n=fuzzy_matching )
            y_pred_sbt = fuzzy_match( y_test, y_pred_sbt, n=fuzzy_matching )
            y_pred_rnd = fuzzy_match( y_test, y_pred_rnd, n=fuzzy_matching )

        # calculate and store accuracies
        acc_dtw.append( accuracy_score(y_test, y_pred_dtw) )
        acc_sbt.append( accuracy_score(y_test, y_pred_sbt) )
        acc_rnd.append( accuracy_score(y_test, y_pred_rnd) )

    print('dtw: ', np.average(acc_dtw), acc_dtw )
    print('sbt: ', np.average(acc_sbt), acc_sbt )
    print('rnd: ', np.average(acc_rnd), acc_rnd )


def prep_data( dataset=15, verify=True ): # verification was OK

    # load dataset with both fdt and qn as subject
    qn, D, y, g, all_types = load( dataset, data_column='q_n', return_depth=True )
    fdt, D_1, y_1, g_1, all_types_1 = load( dataset, data_column='f_dt', return_depth=True )

    if verify: # verify data from loads is equal
        for i, (d1, d2) in enumerate(zip(D, D_1)):
            if (d1!=d2).all():
                print(i, 'Not the same depths', d1, d2) # none found
        for i, (y1,y2) in enumerate(zip(y,y_1)):
            if (y1!=y2).all():
                print(i, 'Not the same labels', y1, y2) # none found
        for i, (g1,g2) in enumerate(zip(g,g_1)):
            if (g1!=g2).all():
                print(i, 'Not the same groups', g1, g2) # none found


    # calculate qns and std(fdt) from 30cm
    qns = []
    std_fdt = []

    # use indexing to calculate qns_30 and std std_30 from any dataset version (apart from 0.2m)
    l_0 = min(13,len(qn[0]))
    offset = int(l_0/2)
    cen_idx = int( len(qn[0])/2 ) + 1

    l_idx = cen_idx - offset - 1
    u_idx = cen_idx + offset

    for some_long_qn, some_long_fdt, some_d in zip(qn, fdt, D):
        #print( some_d )
        #print( some_d[14:27] ) # checked manually -> was as desired
        qns.append( np.average(some_long_qn[l_idx:u_idx]) ) # qns is the 0.3m windowed average
        std_fdt.append( np.std(some_long_fdt[l_idx:u_idx]) ) # same window for std(fdt)

    return qn, np.array(std_fdt), np.array(qns), y, g, all_types


def fuzzy_match( y_test, y_pred, n ): # match soil type with nearest neighbor
    {                     'Clay':   0,  'Silty clay':    1, 
      'Clayey silt':  2,  'Silt':   3,  'Sandy silt':    4,
      'Silty sand':   5,  'Sand':   6,  'Gravelly sand': 7,
      'Sandy gravel': 8,  'Gravel': 9,
    }

    if not isinstance(y_test, np.ndarray): y_test=np.array(y_test)
    if not isinstance(y_pred, np.ndarray): y_pred=np.array(y_pred)

    fuzzy_mask = np.abs( y_pred-y_test) < (n+1)
    y_pred[fuzzy_mask] = y_test[fuzzy_mask]

    return y_pred


def plot_acc_calc( dataset=11, fuzzy_matching=0, f_name='saves/acc_scores.pkl' ):
    labels = {
                ( 15, 0 ): {
                        'f_name':'saves/15_n0_acc_scores.pkl',
                        1: { 
                            'elevation':22.5860966047347, 
                            'txt': ['SBT chart accuracy', (169,0.75), (186,0.82)]
                        },
                        2: { 
                            'pt': (101,0.2),
                            'txt': ['Effective combination', (101, 0.2), ( 146,0.36 )]
                        }
                    },
                ( 15, 1 ):{
                        'f_name':'saves/15_n1_acc_scores.pkl',
                        1:{ 'elevation': 48.597920063854855,
                           #'txt': ['SBT chart accuracy', (169,0.75), (186,0.82)]
                           },
                        2: { 
                            'pt': (101,0.2),
                            'txt': ['Effective combination', (101, 0.2), ( 111,0.425 )]
                        },
                    },
                ( 0, 0 ):{
                        'f_name':'saves/0_n0_acc_scores.pkl',
                        1:{ 'elevation': 67.7,
                           #'txt': ['SBT chart accuracy', (169,0.75), (186,0.82)]
                           },
                        2: { 
                            'pt': (101,0.2),
                            'txt': ['Effective combination', (101, 0.2), ( 111,0.425 )]
                        }
                },
                ( 1, 0 ):{
                        'f_name':'saves/1_n0_acc_scores.pkl',
                        1:{ 'elevation': 67.7,
                           #'txt': ['SBT chart accuracy', (169,0.75), (186,0.82)]
                           },
                        2: { 
                            'pt': (51,0.2),
                            'txt': ['Best DTW classifier', (51, 0.2), ( 70,0.425 )]
                        }
                },
                ( 2, 0 ):{
                        'f_name':'saves/2_n0_acc_scores.pkl',
                        1:{ 'elevation': 67.7,
                           #'txt': ['SBT chart accuracy', (169,0.75), (186,0.82)]
                           },
                        2: { 
                            'pt': (101,0.2),
                            'txt': ['Effective combination', (101, 0.2), ( 111,0.425 )]
                        }
            },
                ( 3, 0 ):{
                        'f_name':'saves/3_n0_acc_scores.pkl',
                        1:{ 'elevation': 67.7,
                           #'txt': ['SBT chart accuracy', (169,0.75), (186,0.82)]
                           },
                        2: { 
                            'pt': (93,0.325),
                            'txt': ['(k=93,w=32.5%) = 67.1% accuracy', (93, 0.325), ( 100,0.525 )]
                        }
                },
                ( 4, 0 ):{
                        'f_name':'saves/4_n0_acc_scores.pkl',
                        1:{ 'elevation': 67.7,
                           #'txt': ['SBT chart accuracy', (169,0.75), (186,0.82)]
                           },
                        2: { 
                            'pt': (101,0.2),
                            'txt': ['Effective combination', (101, 0.2), ( 111,0.425 )]
                        }
                },                
                ( 5, 0 ):{
                        'f_name':'saves/5_n0_acc_scores.pkl',
                        1:{ 'elevation': 67.7,
                           #'txt': ['SBT chart accuracy', (169,0.75), (186,0.82)]
                           },
                        2: { 
                            'pt': (101,0.2),
                            'txt': ['Effective combination', (101, 0.2), ( 111,0.425 )]
                        }
                },
                ( 6, 0 ):{
                        'f_name':'saves/6_n0_acc_scores.pkl',
                        1:{ 'elevation': 67.7,
                           #'txt': ['SBT chart accuracy', (169,0.75), (186,0.82)]
                           },
                        2: { 
                            'pt': (101,0.2),
                            'txt': ['Effective combination', (101, 0.2), ( 111,0.425 )]
                        }
                },
                ( 7, 0 ):{
                        'f_name':'saves/7_n0_acc_scores.pkl',
                        1:{ 'elevation': 67.7,
                           #'txt': ['SBT chart accuracy', (169,0.75), (186,0.82)]
                           },
                        2: { 
                            'pt': (101,0.2),
                            'txt': ['Effective combination', (101, 0.2), ( 111,0.425 )]
                        }
                },
                ( 8, 0 ):{
                        'f_name':'saves/8_n0_acc_scores.pkl',
                        1:{ 'elevation': 67.7,
                           #'txt': ['SBT chart accuracy', (169,0.75), (186,0.82)]
                           },
                        2: { 
                            'pt': (101,0.2),
                            'txt': ['Effective combination', (101, 0.2), ( 111,0.425 )]
                        }
                },
                ( 9, 0 ):{
                        'f_name':'saves/9_n0_acc_scores.pkl',
                        1:{ 'elevation': 67.7,
                           #'txt': ['SBT chart accuracy', (169,0.75), (186,0.82)]
                           },
                        2: { 
                            'pt': (101,0.2),
                            'txt': ['Effective combination', (101, 0.2), ( 111,0.425 )]
                        }
                }
    }

    if 'rnd' in f_name: labels={}

    if (dataset, fuzzy_matching) in labels:
        if 'f_name' in labels[(dataset, fuzzy_matching)]:
            f_name=labels[(dataset, fuzzy_matching)]['f_name']


    fig, ax = plt.subplots( figsize=(6,2.5), tight_layout=True)


    res = load_save( f_name )

    res = { k:v for k,v in res.items() if k[0]==dataset and k[1]==fuzzy_matching }

    if False:
        keep_below = 150
        res = {k:v for k,v in res.items() if k[2]<keep_below}
    
    n=20
    best_n = [key for key, _ in sorted(res.items(), key=lambda x: x[1], reverse=True)[:n]]

    print('best values:')
    for key in best_n:
        print(key, res[key])
    ks, ws, acc = [], [], []

    eps = 1e-3

    for (ds, fm, k, w ), val in res.items():
        if ds==dataset and fm==fuzzy_matching  and w<0.99 and (k in np.arange(1,301,1)) and np.abs(w-0.4)>0.02: #and any(np.abs(np.arange(0,1.01, 0.05) - w) < eps)
            ks.append( k )
            ws.append( w )
            acc.append( val )

    cont = None
    if (dataset, fuzzy_matching ) in labels:
        cont = labels[(dataset, fuzzy_matching )]
    ctitle = 'Accuracy (%)' if 'acc_' in f_name else ('F' + r'$_1$' + '-Measure (%)')
    plot_surface(ax, ks, ws, acc, cont, ctitle=ctitle)

    y_label = 'Window, ' + r'$w$' + ' (% length)'
    y_label = r'$w$' + ', ' + r'$r$' + ' (% length)'
    
    
    ax.set_yticks(np.arange(0,1.01,0.2))    
    labels = [item.get_text() for item in ax.get_yticklabels()]
    labels = [int(float(item.replace('−','-'))*100) for item in labels]
    ax.set_yticklabels(labels)

    ax.set_ylabel( y_label, fontsize=14 )
    ax.set_xlabel( 'Number of neighbors, ' + r'$k$' + ' (-)', fontsize=14 )
    
    x_ticks = np.arange(20,201,20)
    x_ticks[0]=17
    #np.insert(x_ticks,1,17)

    ax.set_xticks(x_ticks)
    ax.set_xlim(1,200)
    
    ax.tick_params( axis='both', labelsize=12 )
    ax.grid(color=(0,0,0))
    
    f_prefix = 'accuracy_' if 'acc_' in f_name else 'f1measure_'
    
    
    plt.tight_layout()
    f_name = f_prefix + '20_' + str(fuzzy_matching) + '.png'
    plt.savefig( f_name, dpi=200, bbox_inches='tight', pad_inches=0.05 ) # dpi=600
    
    #plt.show()


def interpolate_2D( x, y, z, ret_rbf=False ):
    nx = 200
    ny = nx

    z = np.array(z) * 100
    x, y =np.array(x), np.array(y)*100
    
    X = np.column_stack((x,y)) # rbf struggles with different xy scales

    kernel = ['linear','thin_plate_spline','cubic','quintic','gaussian'][1]

    x_min, x_max, y_min, y_max = min(x), max(x), min(0,min(y)), max(max(y),100)
    
    xgrid = np.mgrid[x_min:x_max: nx*1j, y_min:y_max:ny*1j]
    xgrid_out = np.mgrid[x_min:x_max: nx*1j, y_min/100:y_max/100:ny*1j]
    
    rbf = RBFInterpolator(X, z, kernel=kernel)
    xflat = xgrid.reshape(2, -1).T    
    yflat = rbf(xflat)

    ygrid = yflat.reshape(nx, ny)
    if ret_rbf: 
        return xgrid_out, ygrid, rbf
    return xgrid_out, ygrid


def plot_surface( ax, ks, ws ,acc, cont=None, c_label=None, ctitle='' ):
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list( "", ['blue', 'yellow', 'red'] ) #['red','orange','yellow','green','blue']
    xy_mesh, z_mesh,rbf = interpolate_2D( ks, ws, acc,ret_rbf=True )
    #ax.scatter(ks,ws, color=(0,0,0,0), edgecolors=(0,0,0,0.2), s=6, zorder=10)

    
    if False: # ACC_n=0
        norm = matplotlib.colors.Normalize(vmin=16, vmax=32) 
        c_val = 22.39/100

    elif False: # ACC_n=1
        norm = matplotlib.colors.Normalize(vmin=50, vmax=64) 
        c_val = 48.54/100

    elif False: # F1_n=0
        norm = matplotlib.colors.Normalize(vmin=0, vmax=18) 
        c_val = 16.39/100

    else: # F1_n=1
        norm = matplotlib.colors.Normalize(vmin=20, vmax=38) 
        c_val = 37.32/100

    # draw SBT chart result
    ax.contour( *xy_mesh, z_mesh, [c_val*100], linestyles=['-'], linewidths=[3], colors=[ (0,1,1) ] )
    ax.plot([1,150], [-50,-50], ls='-', lw=3, c=(0,1,1), label='SBT Chart: ' + str(round(c_val*100,1)) + '(%)')
    plt.legend( fontsize=12, framealpha=1, loc='upper right' )

    cen_map = ax.pcolormesh( *xy_mesh, z_mesh, cmap=cmap, norm=norm )
    c_plt = ax.contour( *xy_mesh, z_mesh,linestyles=['--'], colors=[(0,0,0,1)] )
    ax.clabel( c_plt, c_plt.levels, inline=True, fontsize=12 )
    cb = plt.colorbar( cen_map ) # draw legend 
    if c_label is None: c_label=ctitle
    
    ###########
    #cb.set_label(c_label, size=14 ) # removed for thesis body
    ###########
    
    # highlighted_point
    ax.scatter( 17, 20/100, s=40, fc=(1,1,1,1), ec=(0,0,0,1), zorder=10)
    
    # and label it
    m_val = rbf( np.array( ([17],[20]) ).T )[0] # rbf value at desired point
    best_label = str( round( m_val,1) ) + '% ' 
    t = ax.annotate(best_label, xy=[17,0.20], xytext=[17+5,0.35], fontsize=12, ha='left', arrowprops=dict(arrowstyle="-", connectionstyle="arc3"), zorder=9)
    t.set_bbox(dict(facecolor=(1,1,1,1), edgecolor=(0,0,0,1)))

    
    cb.ax.tick_params( labelsize=12 )
    ax.set_xlim(0,200)
    ax.set_ylim(0,1)


    #idx = np.argmax( acc )
    #ax.scatter( ks[idx] , ws[idx], color=(1,1,1,1),edgecolors=(0,0,0), s=50, zorder=10)

    # draw and label SBT chart equivalent
    if cont is not None: 
        for k in cont:
            if 'elevation' in cont[k]:
                chrt = ax.contour( *xy_mesh, z_mesh, [cont[k]['elevation']], linewidths=[5], linestyles=['-'], colors=[(92/255,45/255,145/255,1)] )
            if 'txt' in cont[k]:
                ax.annotate(cont[k]['txt'][0], xy=cont[k]['txt'][1], xytext=cont[k]['txt'][2], fontsize=14, arrowprops=dict(facecolor='black', shrink=0.05),)
            if 'pt' in cont[k]:
                pass
                #ax.scatter( cont[k]['pt'][0] , cont[k]['pt'][1], color=(1,1,1,1),edgecolors=(0,0,0), s=50, zorder=10)


if __name__=='__main__':
    eps = 1e-5
    L = 41
    dl = max(0.04, 1/L)
    w = np.arange( 0, 1 + 10*eps, dl ) + eps
    k=np.arange(1, 200, 2)

    #compare_GSA_classifiers( fuzzy_matching=0, dataset=15 )
    for dataset in [ 20 ]:
        for i in [ 1]:#, 1 ]:
            name = 'saves/acc_psi_' + str(dataset) + '_' + str(i) + '_groups.pkl'
            name = 'saves/f1_psi_' + str(dataset) + '_' + str(i) + '_groups.pkl'
            #acc_calc_n( dataset=dataset, w=w, k=k, fuzzy_matching=i, f_name=name )
            plot_acc_calc( dataset=dataset, fuzzy_matching=i, f_name=name )