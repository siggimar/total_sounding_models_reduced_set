import numpy as np
from dtaidistance import dtw
import heapq
import pickle
import os
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


#################################################################
# This is a slightly modified version of the classifiers scritp #
# this one is revised for classification (the other validation) #
#################################################################


class time_series_KNeighborsClassifier():
    '''
    Classifier to do a simple kNN analysis with time series

    The setup is 'impure' as the class sometimes stores both training and testing data.
    The reason behind this design is to store/reuse costly dtw calculations during CV validation.
    Efforts have been made to keep training-/testing data separated in each split.

    Distance metric can be set to either 
      'dtw': ( vanilla dynamic time warping )
      'r': Pearson's correlation coefficient


    followin methods are provided for normal use
      fit( X_train, y_train, X_test ): returns None
      predict( k ): returns y_pred

    '''


    def __init__( self, n_neighbors=11, metric='dtw', load_indices='' ):
        self.distances = {}
        self.update_distances = True
        self.n_neighbors = n_neighbors
        self.dist = self.set_distance_function( metric ) # defaults to dtw
        self.indice_file = load_indices
        self.all_labels = []
        self.load_indices()
        self.set_base_id( np.random.random() ) # not important


    def fit( self, X, y ):
        self.X, self.y = X, y

        if not hasattr(self, 'w'): self.w = 1 # 100% of sequence length 
        self.set_w( self.w ) # update window

        self.all_labels = list( set(y) )
        self.all_labels.sort()


    def fit_( self, idx_train, idx_test ):
        self.idx_train = idx_train
        self.idx_test = idx_test

        # sorted_test_indices has shape=(len(idx_test), len(idx_train)) and contains sorted
        # indices of self.X relative to X_test that are found in X_train.
        # replaced previous dict lookup tables for indexing ( ~3000 times faster: same result)
        self.sorted_test_indices =  self.distances[self.b_id][self.w][ 
            np.isin( self.distances[self.b_id][self.w], idx_train) 
            ].reshape( self.distances[self.b_id][self.w].shape[0], -1)[idx_test]


    def set_base_id( self, base_id ):
        self.b_id = base_id


    def set_w( self, w=1 ):
        self.w = w
        window = max( 1,int(w * len(self.X[0])) ) # window to steps ( not % of length )
        print( 'window set to ' + str(window) + ' units' )
        self.dist.set_window( window ) # set dtw calculation window


    def predict( self, X_test ): # check for distances for X_test: do if not
        self.X_test = X_test

        if self.update_distances: self.calc_distances( self.X_test )
        if isinstance( self.n_neighbors, np.ndarray ): # checks multiple n_neightbors
            return self.predict_n_()
        return self.predict_() # standard


    def predict_( self ): # single n_neighbors        
        y_pred, y_probabilities = self.vote( self.n_neighbors )

        # save return results
        self.probabilities = np.array( y_probabilities )
        return np.array( y_pred )


    def predict_n_( self ):
        y_preds = []
        y_probs = []

        for n in self.n_neighbors:
            y_pred, y_probabilities = self.vote( n )

            y_preds.append( np.array(y_pred) )
            y_probs.append( np.array(y_probabilities) )

        self.probabilities = y_probs
        return np.array( y_preds )


    def calc_distances( self, X_test ): # tested {tuple(some_x):[dwt_dists]}: much better readability but way too slow!
        dist_to_self = False
        
        # modified for classification
        #if np.array_equal(X_test,self.X): dist_to_self=True
        #else: self.distances[self.b_id] = {} #
        
        dist_to_self = False
        self.distances[self.b_id] = {}

        if self.b_id not in self.distances: self.distances[self.b_id] = {}
        if self.w in self.distances[self.b_id]:
            print( 'distances already calculated' )
            return # early out
        self.distances[self.b_id][self.w] = np.empty( shape=(len(X_test), len(self.X)) )


        start = time.time()
        n = int(len( X_test ) / 20) #5%

        print( '\ncalculating distances' )
        for i in range( len(X_test)): # for all data in series
            if i%n==0:  
                t = str(i) + '/' + str( len(X_test) ) + ' - ' + str( round((i+1)/len(X_test)*100, 0) ) + '%'
                t +=  '  (' + str(int(time.time()-start)) + 's)'
                print( t, end='\r')

            if dist_to_self: # utilize symmetry
                for j in range(i, len(self.X)):
                    self.distances[self.b_id][self.w][i][j] = self.dist(X_test[i], self.X[j])
                    self.distances[self.b_id][self.w][j][i] = self.distances[self.b_id][self.w][i][j] # only true if measuring to self! 
            else:
                for j in range( len(self.X) ):
                    self.distances[self.b_id][self.w][i][j] = self.dist(X_test[i], self.X[j])

            # only store sorted indices
            self.distances[self.b_id][self.w][i] = np.argsort( self.distances[self.b_id][self.w][i] )
        self.distances[self.b_id][self.w] = self.distances[self.b_id][self.w].astype(int) # 25% of file size

        if not dist_to_self: self.sorted_test_indices = self.distances[self.b_id][self.w]
        #self.save_indices() # these accumulate fast!
        print( '\nall done... (' + str(round((time.time()-start),1)) + 's)' )


    def vote( self, n_neighbors ):

        y_pred = np.zeros( len(self.X_test) ) - 1
        y_probabilities = np.zeros( shape=(len(self.X_test), len(self.all_labels)) )

        # take a vote for each element
        for i, row in enumerate(self.sorted_test_indices):
            labels = self.y[ row[:n_neighbors] ]

            counts = np.bincount( labels, minlength=len(self.all_labels) )
            label = counts.argmax()

            y_pred[i] = label
            y_probabilities[i] = counts/np.sum(counts) # using sums and not n_neighbors->works weights are added

        return y_pred, y_probabilities


    def n_smallest_keys( self, some_dict, n_neighbors ): # not used
        return heapq.nsmallest( n_neighbors, some_dict, key=some_dict.get )


    def set_distance_function( self, metric ):
        if metric=='r': return self.distance_score_r()
        return self.distance_score_dtw()



    def load_indices( self ):
        if not os.path.isfile( self.indice_file ): return
        with open( self.indice_file, 'rb') as f:
            self.distances = pickle.load( f )


    def save_indices( self ):
        if self.indice_file=='': return
        with open( self.indice_file, 'wb' ) as f:
            pickle.dump( self.distances, f )


    class distance_score_dtw(): # dynamic time warping distance
        def set_window( self, window ):
            self.window = window


        def __call__( self, s_1, s_2 ):
            return dtw.distance_fast( s_1, s_2, window=self.window, psi=self.window ) # 0 to np.inf #, psi=self.window # , use_pruning=True


    class distance_score_r(): # Pearson's correlation coefficient
        def set_window( self, window ): print('\'window\' argument not used in Pearson\'s r') # as with dtw variant
        def __call__( self, s_1, s_2 ):
            return 1-np.corrcoef(s_1, s_2)[0, 1] #-1 to 1


class simple_sens_classifier():
    def __init__( self, threshold=50, apply_bounds=False ):
        self.apply_bounds = apply_bounds
        self.t = threshold


    def set_threshold( self, threshold ):
        self.t=threshold


    def fit( self ):
        pass # intentionally blank


    def predict( self, qns, std_fdt ):
        a = 2/0.0089
        b = -0.61435
        y0 = 0.0006
        x0 = 40

        y = [ 0 ] * len( qns )
        for i, (some_qns, some_std_fdt) in enumerate( zip(qns,std_fdt) ):
            # p as a variable for better readability
            p = a*( np.arctan( np.log10(some_std_fdt/y0) / (np.sqrt((np.log10(some_qns/x0))**2 + (np.log10(some_std_fdt/y0))**2)+np.log10(some_qns/x0)) ) + b)
            if self.apply_bounds: p = max(min(p,100),0)
            y[i] = 1*( p > self.t ) # True -> 1 for p>t
        return y
    
    def P( self, qns, std_fdt ): # added for practical presentation. self.predict() should now utilize self.P()
        a = 2/0.0089
        b = -0.61435
        y0 = 0.0006
        x0 = 40

        p_res = [ 0 ] * len( qns )        
        for i, (some_qns, some_std_fdt) in enumerate( zip(qns,std_fdt) ):
            # p as a variable for better readability
            p = a*( np.arctan( np.log10(some_std_fdt/y0) / (np.sqrt((np.log10(some_qns/x0))**2 + (np.log10(some_std_fdt/y0))**2)+np.log10(some_qns/x0)) ) + b)
            if self.apply_bounds: p = max(min(p,100),0)
            p_res[i] = p
        return p_res

    def class_colors( self, y_test=None ):
        clf_colors = [ colors['Outside model'] ] * len( y_test )

        for i, some_y in enumerate(y_test):
            if some_y == 0 :
                clf_colors[i] = colors['Not sensitive']
            elif some_y == 1 :
                clf_colors[i] = colors['Sensitive']
        return clf_colors


class random_classifier():
    def __init__( self ):
        pass

    def fit( self, X, y ):
        self.all_ys = list( set(y) )
        self.all_ys.sort()

    def predict( self, X_test ):
        return np.random.choice( self.all_ys, len(X_test) )


class tot_dtw_clf( time_series_KNeighborsClassifier ):
    def __init__(self, n_neighbors=11, w=0.2, dataset=15):
        super().__init__( n_neighbors )
        self.w = w


class SBT_chart_classifier():
    def __init__( self, definition ):
        self.name = definition['name']
        self.ref = definition['ref']
        self.var_x = definition['var_x']
        self.var_y = definition['var_y']
        self.log_x = definition['log_x']
        self.log_y = definition['log_y']
        if 'figure' in definition: self.figure = definition['figure']

        self.add_regions( definition['regions'] )

    def predict( self, x, y ):
        is_iter = True
        if not (isinstance(x, list) or isinstance(x, np.ndarray)):
            is_iter = False
            x, y = [x], [y]
        res = [-1] * len(x)

        for i, (some_x, some_y) in enumerate( zip(x, y) ):
            for region in self.regions:
                if region.contains( some_x, some_y ):
                    res[i] = region.nr()
                    break
        if is_iter: return res
        return res[0]


    def fit( self, class_dict ): # fit class_idx to class names
        changed_keys = {}
        for key, value in class_dict.items():
            key_not_found = True
            for region in self.regions:
                if value == region.region_name:
                    changed_keys[value] = ( region.region_nr, key )
                    region.region_nr = key
                    key_not_found=False
            
            if key_not_found:
                print(' unable to match soil class: ' + value)
        
        print( 'SBT classification chart fit to dataset:')
        for key, value in changed_keys.items():
            print( key, 'changed from ' + str(value[0]) + ' to ' + str(value[1]))


    def class_colors( self, y_test=None ):
        
        
        r_colors = {}
        for region in self.regions:
            r_colors[region.region_nr] = region.region_color
        

        colors = [ r_colors[-1] ] * len( y_test )

        for i, some_y in enumerate(y_test):
            if some_y in r_colors: # or_outside def...
                colors[i] = r_colors[ some_y ]
        return colors


    def add_regions( self, region_defs ):
        self.regions = []
        for region in region_defs:
            self.regions.append( SBT_chart_classifier.region(
                region_nr=region, 
                region_def=region_defs[region], 
                log_x=self.log_x, log_y=self.log_y) 
            )


    def test_model( self ):
        x_lims = [np.inf, -np.inf]
        y_lims = [np.inf, -np.inf]

        for r in self.regions:
            if hasattr(r, 'min_x'):
                x_lims[0] = min(x_lims[0], r.min_x)
                x_lims[1] = max(x_lims[1], r.max_x)
                y_lims[0] = min(y_lims[0], r.min_y)
                y_lims[1] = max(y_lims[1], r.max_y)

        n_pts = 250

        eps = 1e-9

        xs = np.logspace( np.log10(x_lims[0]+eps), np.log10(x_lims[1]-eps), n_pts )
        ys = np.logspace( np.log10(y_lims[0]+eps), np.log10(y_lims[1]-eps), n_pts )

        x, y, y_test = [], [], []
        for some_x in xs:
            for some_y in ys:
                x.append(some_x)
                y.append(some_y)
                y_test.append( self.predict(some_x, some_y) )

        y_colors = self.class_colors( y_test )
        y_edge_colors = [ (.2,.2,.2) for y in y_colors ]
        y_edge_colors = y_colors

        fig, ax = self.get_base_fig()
        #self.draw_regions(ax)
        ax.scatter( x, y, c=y_colors, edgecolors=y_edge_colors, s=9, zorder=10 )
        plt.show()

    # presentation functions
    def to_fig( self, x=None, y=None ):
        fig, ax = self.get_base_fig()
        self.draw_regions( ax )
        if x is not None: ax.plot( x, y, marker='o', ls='none', mec=(0,0,0), mfc=(1,1,1), ms=10 )
        plt.show()


    def draw_regions( self, ax ):
        for region in self.regions:
            if not hasattr(region, 'min_x'): continue
            vertices = [ (x_i, y_i) for (x_i,y_i) in zip(region.x,region.y) ]
            ax.add_patch( Polygon(vertices, closed=True, facecolor=colors[region.region_name], zorder=-1) )
            ax.plot( region.x, region.y, c=colors['lines'], lw=1.5, zorder=2 )
            ax.text( region.x_cen, region.y_cen, str(region.region_nr) + ': ' + region.region_name + '\n', horizontalalignment='center',verticalalignment='center' )
            ax.plot( region.x_cen, region.y_cen, marker='o', ms=4, mec=(0,0,0), mfc=colors[region.region_name] )


    def get_base_fig(self):
        fig, ax = plt.subplots( figsize=(14,8), tight_layout=True)
        if hasattr(self, 'figure'):
            ax.set_xlim( self.figure['xlims'] )
            ax.set_ylim( self.figure['ylims'] )
            
            if self.log_x: ax.set_xscale('log')
            if self.log_y: ax.set_yscale('log')
            
            ax.set_xlabel(  self.figure['x'], fontsize=14 )
            ax.set_ylabel(  self.figure['y'], fontsize=14 )
        return fig, ax


    class region():
        def __init__(self, region_nr, region_def, log_x, log_y ):
            self.region_nr = region_nr
            self.region_name = region_def['name']
            self.x = region_def['x']
            self.y = region_def['y']

            self.log_x = log_x
            self.log_y = log_y
            self.region_color = region_def['color']

            self.inc_rpm = region_def['incr_rpm'],# these not implemented!
            self.flush = region_def['flush'],
            self.hammer= region_def['hammer'],

            if self.x:
                self.x_cen, self.y_cen = self.polygon_centroid( self.x, self.y, self.log_x, self.log_y )

                self.max_x = max(self.x)
                self.min_x = min(self.x)
                self.max_y = max(self.y)
                self.min_y = min(self.y)


        def nr( self ):
            return self.region_nr

        def name( self ):
            return self.region_name

        def color( self ):
            return self.region_color

        def get_coords( self ):
            return self.x, self.y


        def x_of_y_by_points(self, y, x1, y1, x2, y2):
            if ( self.log_y and y1>0 and y2>0 ):
                if ( self.log_x and y>0 and x1>0 and x2>0 ):# log-log
                    x = 10 ** ( np.log10( y/y1 ) * np.log10( x1/x2 ) / np.log10( y1/y2 ) + np.log10( x1 ) )
                else: # lin-log
                    x = np.log10( y/y1 ) * ( (x1 - x2) / np.log10(y1/y2) ) + x1
            else:
                if ( self.log_x and y>0 and x1>0 and x2>0 ):# log-lin
                    x = 10 ** ( (y - y1) * np.log10(x1/x2) / (y1 - y2) + np.log10(x1) )
                else: # lin-lin ( log(n) where n<=0 will also default here )
                    x = (y-y1) * ( (x1-x2) / (y1-y2) ) + x1
            return x


        def contains( self, some_x, some_y, rate=None, incr_rpm=None, flush=None, hammer=None ):
            contains = False
            
            n = len(self.x)
            j = n-1

            

            if hasattr(self, 'min_x') and ( some_x < self.min_x or some_x > self.max_x or some_y < self.min_y or some_y > self.max_y ):
                pass # it's outside
            else: # point in poly routine
                for i in range(n):
                    if ( ((self.y[i] > some_y) != (self.y[j] > some_y)) and (some_x < self.x_of_y_by_points(some_y, self.x[i], self.y[i], self.x[j], self.y[j])) ):
                        contains = not contains
                    j=i
            return contains


        # for calculationg region centroid
        def polygon_area( self, x, y ):
            a=0
            for i in range(0, len(x)-1):
                a += x[i]*y[i+1]-x[i+1]*y[i]
            return a/2


        def polygon_centroid( self, x, y, logx=True, logy=True ): # for region label coordinates
            x_ = np.log10(x) if logx else x # account for logarithms
            y_ = np.log10(y) if logy else y

            cx, cy = 0, 0
            for i in range(0, len(x_)-1):
                cx += ( x_[i]+x_[i+1] ) * (x_[i]*y_[i+1]-x_[i+1]*y_[i])
                cy += ( y_[i]+y_[i+1] ) * (x_[i]*y_[i+1]-x_[i+1]*y_[i])

            a = self.polygon_area( x_, y_ )
            cx /= (6*a)
            cy /= (6*a)

            if logx: cx=np.power( 10, cx ) # account for logs
            if logy: cy=np.power( 10, cy )

            return cx, cy


alpha = 1
colors = {
    'lines': (0,0,0,1),
    'Outside model': (255,150,0,1),
    'Clay': ( 0, 160, 190, 1 ),
    'Silty clay': ( 76, 188, 209, 1 ),
    'Clayey silt': ( 154, 110, 188, 1 ),
    'Silt': ( 112, 46, 160, 1 ),
    'Sandy silt': ( 70, 0, 132, 1 ),
    'Silty sand': ( 83, 181, 146, 1 ),
    'Sand': ( 10, 150, 100, 1 ),
    'Gravelly sand': ( 0, 119, 54, 1 ),
    'Sandy gravel': ( 118, 118, 118, 1 ),
    'Gravel': ( 60, 60, 60, 1 ),
    'Sensitive clay': ( 242, 96, 108, 1 ),
    'Sensitive': ( 242, 96, 108, 1 ),
    'Quick clay': ( 242, 96, 108, 1 ),
    'Sensitive silt': ( 251,181,76, 1 ),
    'Brittle': ( 242, 96, 108, 1 ), #'Brittle': ( 251, 181, 56, 1 ),
    'Not sensitive': ( 90, 180, 50, 1 ),
}
colors = { k: (v[0]/255,v[1]/255, v[2]/255, v[3] * alpha) for (k,v) in colors.items() }


tot_2D_meths = [
    { # binary params. (e.g. hammering)- 0:Off / 1:Optional / 2:On
        'name': 'SVV 2016',
        'ref': 'Haugen et al (2016) - A preliminary attempt towards soil classification chart from total sounding',
        'var_x': 'q_ns-0_3',
        'var_y': 'std_fdt-0_3',
        'log_x': True,
        'log_y': True,
        'regions':{
            1: {
                'name':'Quick clay', 
                'x':[ 0.1, 2, 12, 30, 50, 0.1, 0.1 ],
                'y': [ 0.09, 0.09, 0.09, 0.01, 0.001, 0.001, 0.09 ],
                'incr_rpm': 0, 'flush': 0, 'hammer': 0,
                'color': colors['Quick clay']
                },
            2: {
                'name':'Clay',
                'x':[ 0.1, 4, 30, 80, 200, 50, 30, 12, 2, 0.1, 0.1 ],
                'y': [ 3, 1.5, 0.3, 0.05, 0.001, 0.001, 0.01, 0.09, 0.09, 0.09, 3 ],
                'incr_rpm': 0, 'flush': 0, 'hammer': 0,
                'color': colors['Clay'] 
                },
            3: {
                'name':'Silt',
                'x':[ 0.1, 10, 100, 300, 1000, 200, 80, 30, 4, 0.1, 0.1 ],
                'y': [ 100, 12, 1, 0.1, 0.001, 0.001, 0.05, 0.3, 1.5, 3, 100 ],
                'incr_rpm': 0, 'flush': 0, 'hammer': 0,
                'color': colors['Silt']
                },
            4: {
                'name':'Sand',
                'x':[ 0.1, 100, 3000, 10000, 10000, 1000, 300, 100, 10, 0.1, 0.1 ],
                'y': [ 1000, 1000, 30, 1, 0.001, 0.001, 0.1, 1, 12, 100, 1000 ],
                'incr_rpm': 0, 'flush': 0, 'hammer': 0,
                'color': colors['Sand']
                }
            }
    },{
        'name': 'TOT SBT chart',
        'ref': 'Valsson et al. (2024) - A revisit of the 2016 SBT chart by Haugen et al.',
        'var_x': 'q_ns-0_3',
        'var_y': 'std_fdt-0_3',
        'log_x': True,
        'log_y': True,

        'figure': {
            'x':r'$q_{ns}$' + ' (-)',
            'y': 'std(' + r'$F_{DT}$' + ') (kN)',
            'xlims':(1e-1,1e4),'ylims':(1e-3,1e2)
        },

        'regions':{
            -1:{ # default classification
                'name':'Outside model', 
                'x':[],
                'y': [],
                'incr_rpm': 0, 'flush': 0, 'hammer': 0,
                'color': colors['Outside model']},
            1:  {
                'name':'Clay',
                'x':[ 0.1, 8, 9, 90, 4000, 0.1, 0.1],
                'y':[ 0.035, 0.2, 0.04, 0.25, 0.001, 0.001, 0.035],
                'incr_rpm': 0, 'flush': 0, 'hammer': 0,
                'color': colors['Clay']},
            2:  {
                'name':'Silty clay',
                'x':[ 8, 70, 90, 9, 8],
                'y':[ 0.2, 0.5, 0.25, 0.04, 0.2],
                'incr_rpm': 0, 'flush': 0, 'hammer': 0,
                'color': colors['Silty clay']},
            3:  {
                'name':'Clayey silt',
                'x':[ 15, 85, 70, 8, 15],
                'y':[ 1.15, 0.96, 0.5, 0.2, 1.15],
                'incr_rpm': 0, 'flush': 0, 'hammer': 0,
                'color': colors['Clayey silt']},
            4:  {
                'name':'Silt',
                'x':[ 0.1, 15, 8, 0.1, 0.1],
                'y':[ 1.9, 1.15, 0.2, 0.035, 1.9],
                'incr_rpm': 0, 'flush': 0, 'hammer': 0,
                'color': colors['Silt']},
            5:  {
                'name':'Sandy silt',
                'x':[ 0.1, 10, 120, 85, 15, 0.1, 0.1],
                'y':[ 100, 100, 1.2, 0.96, 1.15, 1.9, 100],
                'incr_rpm': 0, 'flush': 0, 'hammer': 0,
                'color': colors['Sandy silt']},
            6:  {
                'name':'Silty sand',
                'x':[ 85, 120, 340, 300, 70, 85],
                'y':[ 0.96, 1.2, 1.8, 0.7, 0.5, 0.96],
                'incr_rpm': 0, 'flush': 0, 'hammer': 0,
                'color': colors['Silty sand']},
            7:  {
                'name':'Sand',
                'x':[ 70, 300, 10000, 10000, 4000, 90, 70],
                'y':[ 0.5, 0.7, 0.08, 0.001, 0.001, 0.25, 0.5],
                'incr_rpm': 0, 'flush': 0, 'hammer': 0,
                'color': colors['Sand']},
            8:  {
                'name':'Gravelly sand',
                'x':[ 340, 10000, 10000, 300, 340],
                'y':[ 1.8, 3, 0.08, 0.7, 1.8],
                'incr_rpm': 0, 'flush': 0, 'hammer': 0,
                'color': colors['Gravelly sand']},
            9:  {
                'name':'Sandy gravel',
                'x':[ 620, 10000, 10000, 340, 620],
                'y':[ 100, 100, 3, 1.8, 100],
                'incr_rpm': 0, 'flush': 0, 'hammer': 0,
                'color': colors['Sandy gravel']},
            10: {
                'name':'Gravel',
                'x':[ 10, 620, 340, 120, 10],
                'y':[ 100, 100, 1.8, 1.2, 100],
                'incr_rpm': 0, 'flush': 0, 'hammer': 0,
                'color': colors['Gravel']}
            },
        }
    ]

class tot_sbt( SBT_chart_classifier ):
    def __init__( self ):
        super().__init__( tot_2D_meths[1] )
    


if __name__=='__main__':
    clf = tot_sbt()

    clf.test_model()

    clf.to_fig()