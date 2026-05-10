import tensorflow as tf


def replace_with_array(model, shared_params):
    layer_list = []
    for layer in model.layer_list:
        layer_list.extend(layer.layer)
    param_names_list = model.param_names_list
    for i, param_names in enumerate(param_names_list):
        for param_name in param_names:
            object.__setattr__(layer_list[i], param_name, shared_params.pop(0))


def replace_with_array_k(model: tf.keras.Model, shared_params: list):
    shared_iter = iter(shared_params)
    for layer in model.layers:
        var_to_attr = {
            id(v): k
            for k, v in layer.__dict__.items()
            if isinstance(v, tf.Variable)
        }
        for var in layer.trainable_weights:
            arr = next(shared_iter)
            attr_name = var_to_attr.get(id(var))
            if attr_name is not None:
                object.__setattr__(layer, attr_name, arr)
