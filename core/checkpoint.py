import os
import torch
import torch.nn as nn


class CheckpointIO(object):
    def __init__(self, fname_template, data_parallel=False, **kwargs):
        os.makedirs(os.path.dirname(fname_template), exist_ok=True)
        self.fname_template = fname_template
        self.module_dict = kwargs
        self.data_parallel = data_parallel

    def register(self, **kwargs):
        self.module_dict.update(kwargs)

    def save(self, step):
        fname = self.fname_template.format(step)
        print('Saving checkpoint into %s...' % fname)
        outdict = {}
        for name, module in self.module_dict.items():
            if isinstance(module, nn.DataParallel):
                outdict[name] = module.module.state_dict()
            elif isinstance(module, nn.Module):
                outdict[name] = module.state_dict()
            elif isinstance(module, torch.optim.Optimizer):
                outdict[name] = module.state_dict()
            else:  # assume module is a nn.Parameter object
                outdict[name] = module.data

        torch.save(outdict, fname)

    def load(self, step):
        fname = self.fname_template.format(step)
        assert os.path.exists(fname), fname + ' does not exist!'
        print('Loading checkpoint from %s...' % fname)
        if torch.cuda.is_available():
            module_dict = torch.load(fname)
        else:
            module_dict = torch.load(fname, map_location=torch.device('cpu'))
            
        for name, module in self.module_dict.items():
            if self.data_parallel:
                module.module.load_state_dict(module_dict[name], False)
            else:
                module.load_state_dict(module_dict[name])
