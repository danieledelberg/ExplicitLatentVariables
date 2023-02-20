import torch
import torch.nn as nn
import numpy as np
import pickle

class ZTracker(nn.Module):
    def __init__(self, zmu, zvar):
        super(ZTracker, self).__init__()
        self.zmu = zmu
        self.zvar = zvar
        # zvals shape: N x Zdim for each
        
        
        zmu_embed = nn.Embedding(zmu.shape[0], zmu.shape[1], sparse=True)
        zmu_embed.weight.data.copy_(zmu)
        zvar_embed = nn.Embedding(zvar.shape[0], zvar.shape[1], sparse=True)
        zvar_embed.weight.data.copy_(zvar)
        
        self.zmu_embed = zmu_embed
        self.zvar_embed = zvar_embed
        
        
    def get_zval(self, ind):
        zmu_val = self.zmu_embed(ind)
        zvar_val = self.zvar_embed(ind)
        return zmu_val, zvar_val
    
    def save(self, out_pkl):
        output_zmu = self.zmu_embed.weight.data.cpu().numpy()
        output_zvar = self.zvar_embed.weight.data.cpu().numpy()

        outputs = (output_zmu, output_zvar)
        
        pickle.dump(outputs, open(out_pkl, 'wb'))