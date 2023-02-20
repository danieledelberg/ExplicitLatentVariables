import torch
import torch.nn as nn
import numpy as np
import pickle

class ZTracker(nn.Module):
    def __init__(self, z_vals):
        super(ZTracker, self).__init__()
        self.zvals = z_vals
    
    def get_z_val(self, ind):
        zval = self.zvals[:, ind, :]
        return zval
    
    def save(self, out_pkl)
        outputs = self.zvals.cpu().numpy()
        pickle.dump(outputs, open(out_pkl, 'wb'))