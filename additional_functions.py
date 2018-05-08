import numpy as np

def xavier_init(size):
    if len(size) == 2:
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = 1. / np.sqrt((in_dim + out_dim) / 2.)
    return xavier_stddev

