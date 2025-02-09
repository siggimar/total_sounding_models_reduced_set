from data_loader import data # training data
import matplotlib.pyplot as plt
from GSA_diagram import GSA_dia
import numpy as np


# Script to showcase all GSA curves in full set. requires training_data .JSON files
# They are not included here. Please see: https://doi.org/10.5281/zenodo.14841538


def get_data():
    accepted = [ 'Leire','Siltig Leire','Leirig Silt', 'Silt', 'Sandig Silt', 'Siltig Sand', 'Sand', 'Grusig Sand', 'Sandig Grus', 'Grus' ]
    res = {}
    res_norm = {}

    all_data = data( 'training_data', 15, read_json=True ) #15 for NGM

    for d in all_data.data:
        some_class = all_data.data[d]['labels']['max two class']
        if some_class in accepted:
            some_curve = { 'x':all_data.data[d]['labels']['gsa_curve']['x'], 'y':all_data.data[d]['labels']['gsa_curve']['y'] }
            some_norm_curve = { 'x':all_data.data[d]['labels']['gsa_curve']['x_norm'], 'y':all_data.data[d]['labels']['gsa_curve']['y_norm'] }
            if some_class in res:
                res[ some_class ].append( some_curve )
                res_norm[ some_class ].append( some_norm_curve )
            else:
                res[ some_class ] = [ some_curve ]
                res_norm[ some_class ] = [ some_norm_curve ]
    return res, res_norm


def get_soil_gsa_range( curves ):
    x_min, x_max = curves[0]['x'].copy(), curves[0]['x'].copy() # iniate with any curve
    for curve in curves:
        for j in range(len(curve['x'])): # just keep highest/lowest vals
            x_min[j] = min(x_min[j],curve['x'][j])
            x_max[j] = max(x_max[j],curve['x'][j])

    all_x_s = np.unique(np.concatenate((x_min, x_max))) # combine all-xs
    y_min = np.interp( all_x_s, x_min, curve['y'] ) # upsample both y_vals
    y_max = np.interp( all_x_s, x_max, curve['y'] )
    
    return all_x_s, y_min, y_max


def plot_curve_ranges():
    gsadia = GSA_dia()
    _, norm_curves = get_data()
    
    for curve in norm_curves:
        range = get_soil_gsa_range( norm_curves[curve] )
        curve_color = gsadia.f_colors[ curve ]
        gsadia.plot_range(range, c=curve_color )


def plot_all_curves():
    gsadia = GSA_dia()
    curves, _ = get_data()
    
    for curve in curves:
        curve_color = gsadia.f_colors[ curve ]
        for some_curve in curves[curve]:
            gsadia.plot_curve(some_curve['x'],some_curve['y'], lw=0.3, c=curve_color, label=curve )
    gsadia.add_legend()


def main():
    if True:
        plot_all_curves()
    else:
        plot_curve_ranges()
    plt.savefig('gsa_curves.png', dpi=150, transparent=False )
    plt.show()
        

if __name__=='__main__':
    main()