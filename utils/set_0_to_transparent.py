import h5py
import numpy as np
file = '/groups/saalfeld/home/heinrichl/Downloads/sample_A_20160501 (1).hdf'
hf = h5py.File(file, 'r+')
clefts = np.array(hf['volumes/labels/clefts'])
clefts[clefts==18446744073709551613] =0
clefts[clefts==0] =18446744073709551613
hf['volumes/labels/clefts'][...] = clefts
hf.close()