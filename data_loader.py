import os
import json
import numpy as np
from sklearn.preprocessing import StandardScaler

root='training_data'

def load( set_nr=0, data_column='f_dt', scale_data=False, f_path=None, return_depth=False, min_leaf=25 ):
    '''
    dtw dataset loader for sensitivity- and SBT classification datasets.  
      set_nr: 0-9: Sensitivity datasets ΔD ∈ [ 0.2, 0.3, 0.4, 0.5, 0.8, 1.0, 1.2, 1.4, 1.8, 2.0 ] m
      set_nr: 10-19: GSA datasets, same ΔD-s

    available columns in sets
    data_col in ['d','f_dt', 'f_dt_lowess_20', 'f_dt_lowess_30', 'f_dt_lowess_40', 'q_n', 'q_ns', 'f_dt_res']
    label_col in [ 'base class', 'max two class', 'full class', 'frost heave class', 'coefficient of uniformity', 'coefficient of curvature' ]

    'f_dt_lowess_X' is the LOWESS smoothed curve with X=delta as a point neigborhood (in cm) - Delta is a stable alternative to frac for variate sounding lengths.
    'f_dt_res' = f_dt-f_dt_lowess_20 - residuals when normalized witht the 20cm LOWESS smoothed curve
    '''

    simplify_labels = True
    min_leaf = 0

    some_dataset = data( root, set_nr, f_path )

    X = [] # data
    D = [] # depth
    y_type = [] # labels
    g = [] # groups

    label_column = 'max two class'

    for data_point in some_dataset.data:
        if data_column not in some_dataset.data[data_point]['data']: continue
        X.append( np.array( some_dataset.data[data_point]['data'][data_column] ) )
        D.append( np.array( some_dataset.data[data_point]['data']['d'] ) )
        y_type.append( some_dataset.data[data_point]['labels'][label_column] )
        g.append( some_dataset.data[data_point]['coordinate group'] )

    X = np.array( X )
    D = np.array( D )
    y_type = np.array( y_type )
    g = np.array( g )

    y, used_classes, m_idx = prep_labels( y_type, simplify_labels=simplify_labels, min_leaf=min_leaf )
    X, D, g, y = X[m_idx], D[m_idx], g[m_idx], y[m_idx] # reduce to selected indices

    # alter output: invert & add info
    used_classes = { v:k.capitalize() for (k,v) in used_classes.items() if k!='remove' }
    used_classes['filename'] = some_dataset.dataset_filename # attach filename

    # scales time series to mean=0 and variance=1
    if scale_data: X = np.transpose( StandardScaler().fit_transform( np.transpose(X) ) )

    if return_depth: return X, D, y, g, used_classes
    return X, y, g, used_classes



def prep_labels( y_type, simplify_labels=True, min_leaf=25 ):

    y_type = np.char.lower( y_type )
    y_type = translate_labels( y_type=y_type.astype('<U32') ) # lowercase english classes (astype crucial. stord as <U12:  'grusig sand'->'gravelly san'!)

    if simplify_labels: # keep used accepted classes - rename rest
        y_type = [ y if y in used_class_def().keys() else 'remove' for y in y_type ]

    if min_leaf>0: # remove classes with too few samples
        y_type = apply_min_leaf( y_type, min_leaf, replacement_class='remove' )

    # used labels and indexes
    used_classes = used_class_def( y_type )

    # apply index by label
    y = np.array( [ used_classes[ some_y ] for some_y in y_type ] )

    # indexes to keep (y not =-1 )
    m_idx = np.where( np.array(y)>=0 )[0]

    return y, used_classes, m_idx


def translate_labels( y_type ):
    dictionary = { # silt=silt && sand=sand,
        'leirig': 'clayey', 'leire': 'clay', 'siltig':'silty', 'sandig': 'sandy', 
        'grusig':'gravelly', 'grus':'gravel', 'steinig': 'stoney', 'stein': 'stone',
        'materiale': 'material'
    }

    for i in range(len(y_type)):
        for key, value in dictionary.items():
            y_type[i] = y_type[i].replace(key, value)

    return y_type


def used_class_def( y_type=[] ):
    '''
    Stores accepted class definitions
      returns 
        full definition if no input is given
        a filtered version matching provided input
    '''

    all_used_types = { # label and index
        'clay':0, 'silty clay':1, 'clayey silt':2, 'silt':3, 'sandy silt':4, 
        'silty sand':5, 'sand':6, 'gravelly sand':7, 'sandy gravel':8, 
        'gravel':9, 'steinig grus':10, 'grusig stein':11, 'stein':12,
        'quick clay':0, 'brittle':1, 'not sensitive':2, 'remove': -1
    }

    for y in y_type:
        if y not in all_used_types.keys():
            all_used_types[y] = max( all_used_types.values() ) + 1

    if len(y_type)>0:
        all_used_types = {k:v for (k,v) in all_used_types.items() if k in y_type}

    return all_used_types


def apply_min_leaf( y_type, min_leaf_size, replacement_class ):
    '''
    Overwrites class values with count<min_leaf_size with replacement_class
    No problem if replacement_class fits criteria, gets overwritten with itself
    '''
    
    # bincount
    unique_labels, counts = np.unique(y_type, return_counts=True)
    small_classes = unique_labels[counts < min_leaf_size]
    
    # replacements
    y_type = np.where(np.isin(y_type, small_classes), replacement_class, y_type).tolist()

    return y_type


def file_name( set_nr ):
    some_dataset = data( root, set_nr, read_json=False )
    return some_dataset.file_path


class data():
    def __init__( self, root, set_nr=0, f_path=None, read_json=True ):
        self.root=root
        self.set_nr = set_nr        
        if f_path is not None: 
            self.file_path = f_path
            self.dataset_filename = os.path.basename( f_path )
        else:self.get_filename()
        if read_json: self.read_from_json()


    def get_filename( self ):
        file_names = os.listdir( self.root )
        some_set_nr = max( 0, min(self.set_nr, len(file_names)-1) )
        if some_set_nr != self.set_nr: 
            print( 'Out of bounds (with \'' + str(self.set_nr) + '\') - using set_nr: ' + str(some_set_nr) )
        self.dataset_filename = file_names[ some_set_nr ]
        self.file_path = os.path.join( self.root, self.dataset_filename )


    def read_from_json( self ):        
        print( 'reading data from ' + self.file_path )

        with open( self.file_path, 'r' ) as f:
            self.data = json.load( f )