import os
from data_loader import data
from GSA_diagram import GSA_dia
import matplotlib.pyplot as plt
import numpy as np


# figure to showcase curve segments in the representative training set.
# plots 1m segments where depth for each groups have been replaced so that
# they plot in the same 1m depth interval


data_folder = 'training_data'
accepted = [ 'Leire','Siltig Leire','Leirig Silt', 'Silt', 'Sandig Silt', 'Siltig Sand', 'Sand', 'Grusig Sand', 'Sandig Grus', 'Grus' ]

def get_data( f_path=None ):
    res = {}
    if f_path==None:
        f_path = os.path.join( 'proposed_model', 'reduced_set_20.json')
    all_data = data( 'training_data', 15, read_json=True, f_path=f_path )

    for d in all_data.data:
        some_class = all_data.data[d]['labels']['max two class']
        if some_class in accepted:
            some_curve = { 'd':all_data.data[d]['data']['d'], 'f_dt':all_data.data[d]['data']['f_dt'], 'q_n':all_data.data[d]['data']['q_n'] }
            if some_class in res:
                res[ some_class ].append( some_curve )
            else:
                res[ some_class ] = [ some_curve ]
    return res


def tot_figure():
    tot_data = get_data()
    gsadia = GSA_dia()
    fig, axs = plt.subplots( 1,3, figsize=(10,5), tight_layout=True)
    dx = 0.9
    dy = 0.08

    for i, l in enumerate( accepted ):
        j = int(i/4)
        k = i % 4

        mat_color = gsadia.f_colors[ l ]
        for i, some_tot in enumerate(tot_data[l]):
            lw=0.4 if i>1 else 2
            axs[j].plot( some_tot['f_dt'], np.array(some_tot['d'])-some_tot['d'][0]+k, lw=lw, c=mat_color )
            #axs[2].plot( some_tot['f_dt'], np.array(some_tot['d']), lw=0.2, c=mat_color, zorder=np.random.randint(10,110) )
        
        tx = axs[j].text(30-dx, k+dy, gsadia.translate[l], fontsize=14, verticalalignment='top', horizontalalignment='right', zorder=200)
        tx.set_bbox(dict(facecolor=(1,1,1,.8), edgecolor=(1,1,1,.8)))


    for i, ax in enumerate(axs):
        if i<2:
            for lvl in np.arange(1,6): ax.plot([0,30],[lvl]*2, lw=1, c=(0,0,0) )
        else:
            for lvl in np.arange(1,3): ax.plot([0,30],[lvl]*2, lw=1, c=(0,0,0) )
        ax.set_xlim( 0,30 )
        if i<2:
            ax.set_yticks( np.arange(0,5,1) )
        else:
            ax.set_yticks( np.arange(0,3,1) )
        ax.grid( c=(.6,.6,.6) )
        ax.set_ylim( 4, 0 )
        ax.set_xlabel('Push force, ' + r'$F_{DT}$' + ' (kN)', fontsize=14)
        ax.tick_params(axis='both', labelsize=12)


    axs[0].set_ylabel('Depths by material type, D' + r'$^{*}$' + ' (m)', fontsize=14)
    for i in [1,2]: axs[i].set_ylabel( 'D' + r'$^{*}$' + ' (m)', fontsize=14)

    for i in range(3):
        t = axs[i].text( -0.15, 1.02, chr( ord('A')+i ), horizontalalignment='center',verticalalignment='top', fontsize=20, transform=axs[i].transAxes )

    #plt.savefig( "fig_6A",dpi=600, transparent=False )
    plt.show()
    a=1

if __name__=='__main__':

    tot_figure()