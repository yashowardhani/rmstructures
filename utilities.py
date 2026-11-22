import types
import numpy as np
import h5py

# plotting stuff
def darken(color, amount=0.6):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

#loads all (or requisite) components of a model
def load_hdf5(filename, datasets=[]):
    out = types.SimpleNamespace()
    with h5py.File(filename, "r") as hf:
        if not len(datasets):
            datasets = list(hf.keys())
        for dataset in datasets:
            setattr(out, dataset, np.array(hf[dataset]))
    return out

#loads to txt file into variables with the same name as columns
def load_dat(filename, verbose=True):
    if verbose: print('loading', filename)
    out = types.SimpleNamespace()
    with open(filename) as f:
        columns = f.readline().split()[1:]
    data = np.loadtxt(filename, dtype=str)
    for i, column in enumerate(columns):
        if isfloat(data[0,i]):  setattr(out, column, data[:,i].astype(float))
        else:                   setattr(out, column, data[:,i])
    return out
