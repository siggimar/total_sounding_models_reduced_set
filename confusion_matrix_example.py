import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import seaborn as sns
from data_loader import load


# script generating confusion matrix examples for, for binary class and multiclass classification.
# figure explaining fuzzy matching concept also produced.


f_size = {
    'fig_title': 20,
    'axis_title': 16,
    'axis_vals': 14,
    'shades': 20,
    'fig_nr':25
}

alpha=0.5
c={
    'TP': ( 93/255, 184/255, 46/255),
    'TN': ( 93/255, 184/255, 46/255),
    'FP': ( 192/255, 0, 0),
    'FN': ( 192/255, 0, 0),
    'gray': ( .4, .4, .4 ),
    'white':( 1, 1, 1),
    'black':(0,0,0),
    'dark':( .2, .2, .2)
}


def pseudo_classifier( labels ):
    '''generates pseudo classification results for visuals - results are given!'''
    all_labels = list(set( labels ))
    threshold = .4

    result = np.zeros( shape=labels.shape )
    for i in range( len(labels) ):
        if np.random.uniform() > threshold:
            result[i] = np.random.choice( all_labels )
        else:
            result[i] = labels[i]
    return result


def confusion_matrix_ax( ax, y_true, y_pred, labels, primary=None, annot=True ):


    cm = confusion_matrix( y_true, y_pred )#, labels=labels )

    if annot: cmap=sns.cubehelix_palette(as_cmap=True)
    else: cmap=cmap = LinearSegmentedColormap.from_list('white_gray',[c['white'], c['gray']])

    sns.heatmap( cm, annot=annot, fmt='g', cbar=False, cmap=cmap, ax=ax )
    format_ax( ax, labels, f_size, annot )

    if primary is None: return

    y_offset = 0.045 * len( labels ) # all counts visible
    primary = max(min(primary, len(labels)-1),0) # allowable range

    rects = []
    rects.append( Rectangle((primary, primary), 1, 1, facecolor=c['TP'], alpha=alpha, zorder=3 ) )
    ax.text(primary+0.5, primary+0.5-y_offset, 'TP', size=f_size['shades'], zorder=9, c=(1,1,1), weight='bold', verticalalignment='center', horizontalalignment='center' )

    if primary>0: # above/left
        rects.append( Rectangle((primary, 0), 1, (primary), facecolor=c['FP'], alpha=alpha, zorder=3 ) ) # FP
        rects.append( Rectangle((0, primary), primary, 1, facecolor=c['FN'], alpha=alpha, zorder=3 ) ) # FN
        rects.append( Rectangle((0, 0), primary, primary, facecolor=c['TN'], alpha=alpha, zorder=3 ) ) # TN

        ax.text(primary/2, primary/2-y_offset, 'TN', size=f_size['shades'], zorder=9, c=(1,1,1), weight='bold', verticalalignment='center', horizontalalignment='center' )
        ax.text(primary+0.5, primary/2-y_offset, 'FP', size=f_size['shades'], zorder=9, c=(1,1,1), weight='bold', verticalalignment='center', horizontalalignment='center' )
        ax.text(primary/2,primary+0.5-y_offset, 'FN', size=f_size['shades'], zorder=9, c=(1,1,1), weight='bold', verticalalignment='center', horizontalalignment='center' )

    if primary<len(labels)-1: # below/right
        n_long = (len(labels)-primary)

        rects.append( Rectangle((primary, primary+1), 1, n_long, facecolor=c['FP'], alpha=alpha, zorder=3 ) )
        rects.append( Rectangle((primary+1, primary), n_long, 1, facecolor=c['FN'], alpha=alpha, zorder=3 ) )
        rects.append( Rectangle((primary+1, primary+1), n_long, n_long, facecolor=c['TN'], alpha=alpha, zorder=3 ) ) # TN

        ax.text(primary+0.5, primary+(n_long+1)/2-y_offset, 'FP', size=f_size['shades'], zorder=9, c=(1,1,1), weight='bold', verticalalignment='center', horizontalalignment='center' )
        ax.text(primary+(n_long+1)/2, primary+0.5-y_offset, 'FN', size=f_size['shades'], zorder=9, c=(1,1,1), weight='bold', verticalalignment='center', horizontalalignment='center' )
        if len(labels)>2: ax.text(primary+(n_long+1)/2, primary/2-y_offset, 'TN', size=f_size['shades'], zorder=9, c=(1,1,1), weight='bold', verticalalignment='center', horizontalalignment='center' )
        ax.text(primary+(n_long+1)/2, primary+(n_long+1)/2-y_offset, 'TN', size=f_size['shades'], zorder=9, c=(1,1,1), weight='bold', verticalalignment='center', horizontalalignment='center' )
        if len(labels)>2: ax.text(primary/2,primary+(n_long+1)/2-y_offset, 'TN', size=f_size['shades'], zorder=9, c=(1,1,1), weight='bold', verticalalignment='center', horizontalalignment='center' )

        if primary>0:
            rects.append( Rectangle((0, primary+1), primary, n_long, facecolor=c['TN'], alpha=alpha, zorder=3 ) ) # TN (left bottom)
            rects.append( Rectangle((primary+1, 0 ), n_long, primary, facecolor=c['TN'], alpha=alpha, zorder=3 ) ) # TN

    for j in range(2):
        ax.plot( [primary+j, primary+j], [0,len(labels)], lw=4, c=(1,1,1), zorder=8 )
        ax.plot( [0,len(labels)], [primary+j, primary+j], lw=4, c=(1,1,1), zorder=8 )

    for rect in rects:
        ax.add_patch( rect )

def format_ax( ax, labels, f_size, annot ):
    scale = 1
    if not annot:
        scale *= 0.8

    ax.xaxis.set_ticklabels( labels, fontsize=f_size['axis_vals']*scale )
    ax.yaxis.set_ticklabels( labels, fontsize=f_size['axis_vals']*scale )

    ax.set_xlabel( 'Predicted labels', fontsize=f_size['axis_title'] )
    ax.set_ylabel( 'True labels', fontsize=f_size['axis_title'] )


def get_data( n=2500 ):
    labels = np.array( [ 'clay', 'silt', 'sand', 'gravel', 'stone' ] )
    k = len(labels)

    # generate data and some classification results
    y_true = np.random.randint( 0, k, n*k )
    y_pred = pseudo_classifier( y_true )

    return y_true, y_pred, labels


def simplify( y_true, y_pred, labels, positive=0, negative=1 ):
    yt_mask = np.logical_or(y_true==positive, y_true==negative) # where pos/neg are found
    yp_mask = np.logical_or(y_pred==positive, y_pred==negative) # in either array
    final_mask = np.logical_and(yt_mask, yp_mask) # they're combination
    labels = [labels[i] for i in range(len(labels)) if (i==positive or i==negative)]

    return y_true[final_mask], y_pred[final_mask], labels


def ex_a():
    y_true, y_pred, labels = get_data( n=2500 )
    y_true_, y_pred_, labels_ = simplify(y_true, y_pred, labels, positive=1, negative=3)

    fig, axs = plt.subplots( 1, 2, figsize=( 15, 5 ))
    confusion_matrix_ax( axs[0], y_true, y_pred, labels, primary=1)
    confusion_matrix_ax( axs[1], y_true_, y_pred_, labels_, primary=0)

    fig.text(0.01, 0.94, 'A', fontsize=25, weight='bold', color=(0,0,0), verticalalignment='center', horizontalalignment='left')
    fig.text(0.51, 0.94, 'B', fontsize=25, weight='bold', color=(0,0,0), verticalalignment='center', horizontalalignment='left')

    plt.tight_layout(w_pad=4)
    plt.show()



def ex_b():
    fig, axs = plt.subplots( 1, 2, figsize=( 15, 6 ))
    qn, y, g, all_types = load( 15, data_column='q_n' )

    labels = np.array([ v.replace(' ', '\n') for k,v in all_types.items() if not isinstance(k, str) ])
    y_true_1, y_true_2 = [], []
    y_pred_1, y_pred_2 = [], []
    
    if True: # random classifier accuracy estimate
        a = np.bincount(y)
        a = a/np.sum(a)
        print( a )
        print( np.sum(a) )

        b = [1/10] * 10 # exact matching
        #b = [2/10] + [3/10] * 8 + [2/10] # fuzzu matching n=1
        b = np.array( b )
        preds = np.multiply( a, b )
        print( preds )
        print( np.sum(preds) )

    for i, l in enumerate(labels):
        scale = 1+i%2
        for j in range(scale):
            for s in range(3):
                y_true_1.append( i )
                y_pred_1.append( i )
            
        
            if i>0:
                y_pred_2.append( i-1 ) # one below
                y_true_2.append( i )
            y_pred_2.append( i )
            y_true_2.append( i )
            if i<(len(labels)-1):
                y_pred_2.append( i+1 ) # one above
                y_true_2.append( i )

    confusion_matrix_ax( axs[0], y_true_1, y_pred_1, labels, primary=None, annot=False)
    confusion_matrix_ax( axs[1], y_true_2, y_pred_2, labels, primary=None, annot=False)

    for i in [0,1]:        
        axs[i].set_yticklabels(labels, rotation = 0)

    for i in range(len(labels)+1):
        for j in [0,1]:
            axs[j].plot([i,i],[0,len(labels)], c=c['dark'], lw=1)
            axs[j].plot([0,len(labels)],[i,i], c=c['dark'], lw=1)
    
    fig.text(0.01, 0.94, 'A', fontsize=25, weight='bold', color=c['black'], verticalalignment='center', horizontalalignment='left')
    fig.text(0.51, 0.94, 'B', fontsize=25, weight='bold', color=c['black'], verticalalignment='center', horizontalalignment='left')
    
    plt.tight_layout(w_pad=4)
    plt.show()



if __name__ =='__main__':
    ex_a() # confusion matrix,  multilabel/ binary
    ex_b() # confusion matrix, fuzzy accuracy
