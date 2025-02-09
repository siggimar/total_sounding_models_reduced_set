import os
import numpy as np
import matplotlib.pyplot as plt
import knn_trainer
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from classifiers import time_series_KNeighborsClassifier
from knn_res_plotter import knn_plotter

saves_folder = 'saves'

def train_model( var, cv, classifier, normalize ):
    t_norm = '_normalized' if normalize else ''
    clf_ind_file = os.path.join(saves_folder, 'knn_dtw_' + var + t_norm + '_indices.pkl' )
    eps = 1e-5

    k = np.arange( 1, 200, 2 )

    trainer = knn_trainer.trainer( 
        saves_folder, 
        cross_validation_func=cv( n_splits=2 ),
        classifier=classifier( n_neighbors=k, load_indices=clf_ind_file ),
        var=var,
        n_neighbors=k,
    )

    trainer.load_results()

    for d_set_nr in [ 10 ]:
        print('Training on dataset ' + str(d_set_nr))
        trainer.set_data( d_set_nr, normalize )
        
        L = len(trainer.X[0])
        dl = max(0.04, 1/L)

        w = np.arange( 0, 1 + 10*eps, dl ) + eps


        trainer.validate( w=w, k=k )
        trainer.save_results()


def plot_results( var, d_set ):
    d_set_b = min(9,d_set)
    d_set_a = d_set_b+10

    plotter = knn_plotter()
    trainer = knn_trainer.trainer( 
        saves_folder, 
        cross_validation_func=None,
        classifier=None,
        var=var,
        n_neighbors=None,
    )
    trainer.load_results()

    plotter.comparison_plot( trainer, d_set_a, d_set_b )


if __name__=='__main__':
    data_col = [ 'f_dt', 'q_ns', 'q_n', 'q_n_norm','q_n_psi' ][4]
    normalize=False

    cv = [ StratifiedGroupKFold, GroupKFold ][0]
    classifier = time_series_KNeighborsClassifier

    #train_model( var=data_col, cv=cv, classifier=classifier, normalize=normalize )
    #for i in range(10):
        
    #for i in range(10):
    plot_results( var=data_col, d_set=5)