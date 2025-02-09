import os
import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from classifiers import time_series_KNeighborsClassifier
import knn_trainer
from knn_res_plotter import knn_plotter

saves_folder = 'saves'

def train_model( var, cv, classifier):

    clf_ind_file = os.path.join(saves_folder, 'knn_dtw_reduced_indices.pkl' )
    eps = 1e-5

    k = np.arange( 1, 200, 2 )
    trainer = knn_trainer.trainer( 
        saves_folder, 
        cross_validation_func=cv( n_splits=10 ),
        classifier=classifier( n_neighbors=k, load_indices=clf_ind_file ),
        var=var,
        n_neighbors=k,
    )
    trainer.load_results()

    for d_set_nr in [ 20 ]:
        print('Training on dataset ' + str(d_set_nr))
        d_f_name = 'reduced_set_' + str(d_set_nr) + '.json'
        d_folder = 'proposed_model'
        f_path = os.path.join( d_folder, d_f_name)

        
        trainer.set_data( d_set_nr, normalize=False, f_path=f_path )  # added f_name

        L = len(trainer.X[0])
        dl = max(0.04, 1/L)
        w = np.arange( 0, 1 + 10*eps, dl ) + eps
        trainer.validate( w=w, k=k )
        trainer.save_results()

def plot_results( var, d_set ):
    d_set_b = 70
    d_set_a = 20

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

def main():
    #train_data, test_data = get_data()

    cv = [ StratifiedGroupKFold, GroupKFold ][0]
    classifier = time_series_KNeighborsClassifier

    #train_model( var='q_n', cv=cv, classifier=classifier )
    plot_results( var='q_n', d_set=5)

    a=1

if __name__=='__main__':
    main()