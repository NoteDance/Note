def replace_array(model, shared_params):
    layer_list = []
    for layer in model.layer_list:
        layer_list.extend(layer.layer)
    param_names_list = model.param_names_list
    for i, param_names in enumerate(param_names_list):
        for param_name in param_names:
            object.__setattr__(layer_list[i], param_name, shared_params.pop(0))