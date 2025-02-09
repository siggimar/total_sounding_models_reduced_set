import os
import json
import copy


# script to reduce size of .JSON file with training data.

def reduce_files(): # copies only files marked as part of reduced set (group ID 201)
    d_folder = 'proposed_model'
    f_names = [ 'full_set.json', 'reduced_set_20.json', 'reduced_set_70.json'] # f_names[0] now removed

    f_s = [ os.path.join(d_folder, f) for f in f_names ]
    f_s_out = [ os.path.join( d_folder, 'red_' + f ) for f in f_names ]

    for f, fo in zip( f_s, f_s_out ):
        with open( f, 'r' ) as f:
            res = json.loads(f.read())
        
        trim = False
        for k in res:
            if res[k]['coordinate group']==201:
                trim=True
                break

        if trim: res = { k:v for k,v in res.items() if v['coordinate group']==201}
        
        res_out = {}

        for k in res:
            tmp_res = {}
            tmp_res['description'] = copy.deepcopy( res[k]['description'] )
            tmp_res['coordinates'] = copy.deepcopy( res[k]['coordinates'] )
            tmp_res['coordinate group'] = copy.deepcopy( res[k]['coordinate group'] )
            
            tmp_res['data'] = {}
            for tmp_key in ['d', 'f_dt', 'q_n']:
                tmp_res['data'][tmp_key] = copy.deepcopy( res[k]['data'][tmp_key] )
            
            tmp_res['labels'] = copy.deepcopy( res[k]['labels'])
            del tmp_res['labels']['gsa_curve']['x_norm']
            del tmp_res['labels']['gsa_curve']['y_norm']

            res_out[k] = tmp_res
        
        with open( fo, 'w' ) as f:
            f.write( json.dumps( res_out, indent=4 ) )


if __name__=='__main__':
    reduce_files()