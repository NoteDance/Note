from Note import nn


class LayerMeta(type):
    def __call__(cls, *args, **kwargs):
        # Create the instance
        instance = super().__call__(*args, **kwargs)
        # Process any Parameter assignments that occurred during __init__
        instance._finalize_parameters()
        return instance


class Layer(metaclass=LayerMeta):
    def __init__(self):
        # store this layer’s own variables
        self._own_params = []
        # store any sub‑layers you attach
        self._sub_layers = []
        
        self._param_assignments = []
        
        self.name_ = self.__class__.__name__
        
        self.layer_list = []
        
        if hasattr(self, 'init_weights'):
            nn.Model.add()
            if len(nn.Model.name_list)>0:
                nn.Model.name_=nn.Model.name_list[-1]
            if nn.Model.name_ != None and nn.Model.name_ not in nn.Model.layer_dict:
                nn.Model.layer_dict[nn.Model.name_] = []
                nn.Model.layer_dict[nn.Model.name_].append(self)
            elif nn.Model.name_ != None:
                   nn.Model.layer_dict[nn.Model.name_].append(self)
    
    def _finalize_parameters(self):
        # Process any parameter assignments that were deferred
        for param in self._param_assignments:
            self._own_params.append(param)
        self._param_assignments.clear()
    
    def add_param(self, var):
        if hasattr(self, '_param_assignments'):
            self._param_assignments.append(var)
        else:
            self._own_params.append(var)

    def __setattr__(self, name, value):
        # whenever you assign an nn.Layer (or Layer) to an attribute,
        # register it as a child
        if isinstance(value, Layer):
            # add to our children list
            object.__getattribute__(self, "_sub_layers").append(value)
            object.__setattr__(value, 'name', name)
            self.layer_list.append(value)
        object.__setattr__(self, name, value)
    
    @property
    def param(self):
        # recursively collect all own and sub‑layers’ parameters
        out = list(self._own_params)
        for child in self._sub_layers:
            out.extend(child.param)
        return out
