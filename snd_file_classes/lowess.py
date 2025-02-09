import numpy as np
import warnings

class lowess():
    '''
    Lowess class based on methods described in Cleveland (1979): "Robust Locally Weighted Regression and Smoothing Scatterplots"

    Weighted polynomials calculated using numpy.

    Standard parameter f (or frac) exchanged for depth range parameter delta.
    i.e. fits are calculated using points within distance delta and not some closest % of total points.

    Implementation inspired by lowess.py code provided by Alexandre Gramfort on github
    '''
    def __init__( self, delta, deg=2, iterations=3 ):
        self.short_name = 'lowess'
        self.long_name = 'Robust Locally Weighted Regression'
        self.delta = delta
        self.w_bound = 1e-9 # ensure weights are never exactly 0
        self.deg = deg
        self.iterations = iterations
        warnings.simplefilter('ignore', np.RankWarning)


    def get_short_name( self ):
        return self.short_name + ' (delta=' + str(self.delta) + ', deg=' + str(self.deg) + ', it=' + str(self.iterations) +')'


    def tri_cubic( self, x ): # tricube function
        x_i = np.clip( np.abs(x), 0, 1-self.w_bound ) # ensure tri(n) ϵ [0,1]
        return ( 1 - x_i**3 )**3


    def bi_square( self, x ): # bisquare function
        x_i = np.clip( x, -1+self.w_bound, 1-self.w_bound ) # ensure bi(n) ϵ [0,1]
        return ( 1 - x_i**2 )**2


    def weighted_regression( self, x_k, x, y, weights ): # np function utilized for regression
        coeff = np.polyfit( x, y, deg=self.deg, w=weights )
        return np.polyval( coeff, x_k )


    def max_abs_residuals( self ):
        m_eps = np.zeros( shape=self.eps.shape )
        for i in range( len(self.x) ):
            abs_local_eps = np.abs( self.eps[ self.idx[i] ] )
            m_eps[i] = np.max( abs_local_eps )
        return m_eps


    def std_abs_residuals( self ):
        m_eps = np.zeros( shape=self.eps.shape )
        for i in range( len(self.x) ):
            local_eps = self.eps[ self.idx[i] ]
            abs_local_eps = np.abs( local_eps )
            m_eps[i] = np.std( local_eps )
        return m_eps


    def fit( self, x, y ): # x & y are 1D np.arrays
        self.x = np.array( x )
        self.y = np.array( y )

        self.idx = []
        self.xi  = []
        self.yi  = []
        self.w   = []
        self.y_pred_0 = []
        h   = [] 

        for i in range(len(self.x)):
            self.idx.append( np.asarray( np.abs(self.x - self.x[i]) < self.delta ).nonzero()[0] ) # indexes of x for each x0 where |x-x0|<d
            self.xi.append( self.x[ self.idx[-1] ] )
            self.yi.append( self.y[ self.idx[-1] ] )

            # calculate x-weights (done once)
            h.append( np.max( np.abs(self.xi[-1] - self.x[i]) ) )
            x_rel = ( self.xi[-1]-self.x[i] ) / h[-1]
            self.w.append( self.tri_cubic(x_rel) )

            self.y_pred_0.append( self.weighted_regression(self.x[i], self.xi[-1], self.yi[-1], weights=self.w[-1]) )
        self.y_pred_0 = np.array(self.y_pred_0) # keep locally weighted regression


    def predict( self, _=None ): # predicted on fitted x
        y_pred = self.y_pred_0.copy() # start with locally weighted
        for it in range( self.iterations ):
            eps = self.y - y_pred   # all prediction residuals
            s = np.median( np.abs( eps ) )
            delta = self.bi_square( eps/(6*s) )

            for i in range( len(self.x) ):
                idx = self.idx[i] # local point id
                d_idx = delta[idx]
                if d_idx.max()==0 and d_idx.min()==0:
                    d_idx = np.ones(shape=d_idx.shape) * self.min_delta
                delta_wi = np.multiply( self.w[i], d_idx ) # update local weights
                y_pred[i] = self.weighted_regression( self.x[i], self.xi[i], self.yi[i], weights=delta_wi )

        self.eps = self.y - y_pred # keep residuals for analysis
        return y_pred # return robust locally weighted regression