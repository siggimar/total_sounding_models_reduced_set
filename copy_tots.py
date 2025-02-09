from tqdm import tqdm
import numpy as np
import os
import shutil
from snd_file_classes.snd_files import snd_file

'''
this script was written to copy random examples of total sounding files from the project archive

criteria were added to only include files 
  having ~25mm aquisition increment (or less)
  just under 10m of data (selected for the paper presentation)
  soundings should start close to surface

The snd filetype is used (rather than the .tot or .std raw files) as a class was available to 
parse it (from the data mining phase of the project). This format was originally selected as 
it contained sufficiently good sounding data along with coordinates in the same file.
'''

base_path = '/GuDb_data'
to_path = 'example_files'
from_depth_less_than = 0.2 #m
to_depth = 8 #m
dy = 0.3
min_increment = 0.026
n = 200

def copy_example_data():
    files = [file for root, dirs, files in os.walk(base_path) for file in [os.path.join(root, f) for f in files]]
    files_ = [f for f in files if '.snd' in f.lower()]
    np.random.shuffle( files_ ) # this ensures files from across Norway

    snd_s = []

    for i in tqdm(range(len(files_))):
        f = files_[i]
        if '.snd' in f.lower() and len(snd_s)<n:
            some_file = snd_file( f ) # read file
            some_file.parse_raw_data()

            if not len(some_file.soundings['tot'])>0: continue # early abort
            some_tot =  some_file.soundings['tot'][0] # only check first

            increment = some_tot.data[0][1]-some_tot.data[0][0]
            if abs(increment)>min_increment: continue
            delta_d = some_tot.data[0][-1] - some_tot.data[0][0]
            if delta_d > (to_depth-dy) and delta_d<to_depth:
                if some_tot.data[0][0] < from_depth_less_than and some_tot.data[0][0] > 0:
                    to_file = os.path.join(to_path, os.path.basename(f))
                    snd_s.append( (f, to_file) )
                    print(len(snd_s))

        if len(snd_s)==n: break

    for snd_tuple in snd_s: # finally copy selected files
        shutil.copyfile( snd_tuple[0], snd_tuple[1] )


if __name__=='__main__':
    copy_example_data() # runtime 8 mins 7 sec