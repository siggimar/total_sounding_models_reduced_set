import os
import numpy as np
from data_loader import load
import pickle
from CV_method_choice import sort_data

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from sklearn.preprocessing import LabelBinarizer

'''
This module is written to perform CV and hyperparameter tuning for knn with dtw in a custom class
'''


class trainer():
    def __init__( self, saves_folder, cross_validation_func, classifier, var, n_neighbors ):
        if not os.path.isdir(saves_folder): os.mkdir( saves_folder )
        self.saves_folder = saves_folder
        self.cv = cross_validation_func
        self.classifier = classifier
        self.var = var
        self.n_neighbors = n_neighbors
        self.results = {}


    def set_data( self, d_set_nr, normalize, f_path ):        
        self.d_set = d_set_nr
        self.X, self.y, self.g, self.all_types = load( d_set_nr, data_column=self.var, scale_data=normalize, f_path=f_path )


    def validate( self, w, k ):
        base_id = self.get_base_id()
        if base_id not in self.results: self.results[base_id] = { 'description': self.all_types['filename'] }

        for some_w in w: # different windows
            print( 'setting w=' + str(some_w) )
            temp_AUC = [ [] for l in range(len( self.n_neighbors )) ] # empty list for each k in k-NN

            self.classifier.set_base_id(self.get_base_id() )
            self.classifier.fit( self.X, self.y )
            self.classifier.set_w ( some_w ) # discards old distances
            self.classifier.calc_distances( self.X ) # consider all X as X_test, have to be careful now
            self.classifier.update_distances = False # unnescessary for validation

            for ii, (tr, tt) in enumerate( self.cv.split(X=self.X, y=self.y, groups=self.g) ):
                print( 'working on split ' + str(ii+1) + '/' + str(self.cv.n_splits), end='\r' )
                if len(tr)>len(tt):
                    #print('skipping n(train)>n(test)')
                    continue
                
                X_train, y_train, X_test, y_test = self.X[tr], self.y[tr], self.X[tt], self.y[tt]

                print( set( y_test ) )

                #self.classifier.fit( X_train, y_train )                
                self.classifier.fit_( idx_train=tr, idx_test=tt ) # keeps full set stored as classifier.X and classifier.y
                y_pred = self.classifier.predict( X_test )
                y_probs = self.classifier.probabilities

                if False:
                    label_binarizer = LabelBinarizer().fit( y_train )
                    y_onehot_test = label_binarizer.transform( y_test )

                    # check outputs
                    display = RocCurveDisplay.from_predictions(
                        y_onehot_test[:, 2],
                        y_probs[10][:, 2],
                        name=f"2 vs the rest",
                        color="darkorange",
                        #plot_chance_level=True,
                    )
                    _ = display.ax_.set(
                        xlabel="False Positive Rate",
                        ylabel="True Positive Rate",
                        title="One-vs-Rest ROC curves:\n2 vs (0 & 1)",
                    )
                    plt.show()


                # roc_auc_score struggles with: missing labels in test set.
                y_prob_labels = np.arange(len(y_probs[0][0]))
                y_test_labels = np.array(list(set(y_test)))
                columns_to_drop = np.setdiff1d( y_prob_labels, y_test_labels ) # find out what is missing
                if columns_to_drop: 
                    for i in range( len(self.n_neighbors) ): # for each case of n_neighbors
                        tmp =  np.delete( y_probs[i], columns_to_drop, axis=1 ) # drop columns missing label columns
                        
                        for j, row in enumerate( tmp ):
                            if np.sum(row)==0: # ie. only prediction got deleted [0,0,0,0,1,0] -> [0,0,0,0,0]
                                tmp[j][np.random.randint(0,len(row))]=1 # make a random guess of the others

                        y_probs[i] = tmp/tmp.sum(axis=1, keepdims=True) # rebuild probabilities (sum=1)


                for i, k in enumerate(self.n_neighbors):
                    temp_AUC[i].append(
                        roc_auc_score( # multiclass ovo CV (ref. Hand & Till 2001)
                            y_test,
                            y_probs[i],
                            multi_class="ovo",
                            average="macro",
                        )
                    )
            print('\nall done.')
            for i, k in enumerate(self.n_neighbors):
                self.results[base_id][(some_w,k)] = np.average( temp_AUC[i] )

    
    def get_results( self, d_set ):
        ws, ks, AUCs = [], [], []
        for key in self.results[d_set].keys():
            if key=='description':continue
            ( w, k ) = key
            ws.append(w)
            ks.append(k)
            AUCs.append(self.results[d_set][ (w, k) ])
        return ws, ks, AUCs


    def plot_results( dataset ):
        pass


    def get_base_id( self ):
        return self.d_set


    def get_savepath( self ):
        return os.path.join( self.saves_folder, self.var + '.pkl' )


    def save_results( self ):
        save_path = self.get_savepath()
        if not self.results: return

        with open( save_path, 'wb' ) as f:
            pickle.dump( self.results, f )


    def load_results( self ):
        save_path = self.get_savepath()
        if not os.path.isfile( save_path ): return

        with open( save_path, 'rb' ) as f:
            self.results = pickle.load( f )