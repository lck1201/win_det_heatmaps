import numpy as np

def float2int(val):
    return tuple(int (x+0.5) for x in val)