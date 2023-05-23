import torch
from torch import nn

class MhcModel(nn.Module):
    
    def __init__(self, model_id="default_model", comment=""):
        super(MhcModel, self).__init__()
        self._main_pars = {}
        self._add_pars = {}
        self._comment = comment
        self._id = model_id


    def name(self):
        return self._id + \
            "," + \
            ",".join([str(key) + "=" + str(val) for key,val in sorted(self._main_pars.items(), key=lambda x: str(x[0]))]) + \
            "," + \
            ",".join([str(key) + "=" + str(val) for key,val in sorted(self._add_pars.items(), key=lambda x: x[0])]) + \
            "," + self._comment


    def save(self, foldername=None):
        pass
        # create folder
        # create txt file with parameters
        # create png plot with structure?
        # save via http://pytorch.org/docs/master/notes/serialization.html#recommend-saving-models
        # viz graph: https://github.com/szagoruyko/pytorchviz