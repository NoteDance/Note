import tensorflow as tf
from Note.nn.initializer import initializer
from Note.nn.activation import activation_dict
from Note.nn.Module import Module
import re


class einsumdense:
    """A layer that uses `tf.einsum` as the backing computation.

    This layer can perform einsum calculations of arbitrary dimensionality.

    Args:
      input_shape: Shape of the input tensor.
      equation: An equation describing the einsum to perform. This equation must
        be a valid einsum string of the form `ab,bc->ac`, `...ab,bc->...ac`, or
        `ab...,bc->ac...` where 'ab', 'bc', and 'ac' can be any valid einsum
        axis expression sequence.
      output_shape: The expected shape of the output tensor (excluding the batch
        dimension and any dimensions represented by ellipses). You can specify
        None for any dimension that is unknown or can be inferred from the input
        shape.
      activation: Activation function to use. If you don't specify anything, no
        activation is applied.
      bias_axes: A string containing the output dimension(s) to apply a bias to.
        Each character in the `bias_axes` string should correspond to a
        character in the output portion of the `equation` string.
      kernel_initializer: Initializer for the `kernel` weights matrix.
      bias_initializer: Initializer for the bias vector.
    """

    def __init__(
        self,
        equation,
        output_shape,
        input_shape=None,
        activation=None,
        bias_axes=None,
        weight_initializer="Xavier",
        bias_initializer="zeros",
        trainable=True,
        dtype='float32'
    ):
        self.equation = equation
        if isinstance(output_shape, int):
            self.partial_output_shape = [output_shape]
        else:
            self.partial_output_shape = list(output_shape)
        self.bias_axes = bias_axes
        if activation is not None:
            self.activation = activation_dict[activation]
        else:
            self.activation = None
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.trainable=trainable
        self.dtype=dtype
        self.input_shape=input_shape
        if input_shape is not None:
            shape_data = _analyze_einsum_string(
                self.equation,
                self.bias_axes,
                input_shape,
                self.partial_output_shape,
            )
            kernel_shape, bias_shape, self.full_output_shape = shape_data
            self.param=[]
            self.weight = initializer(
                shape=kernel_shape,
                initializer=self.weight_initializer,
                dtype=dtype,
            )
            self.param.append(self.weight)
            
            if bias_shape is not None:
                self.bias = initializer(
                    shape=bias_shape,
                    initializer=self.bias_initializer,
                    dtype=dtype,
                )
                self.param.append(self.bias)
            else:
                self.bias = None
            
            if trainable==False:
                self.param=[]
            Module.param.extend(self.param)


    def output(self, data):
        if self.input_shape is None:
            input_shape=data.shape
            shape_data = _analyze_einsum_string(
                self.equation,
                self.bias_axes,
                input_shape,
                self.partial_output_shape,
            )
            kernel_shape, bias_shape, self.full_output_shape = shape_data
            self.param=[]
            self.weight = initializer(
                shape=kernel_shape,
                initializer=self.weight_initializer,
                dtype=self.dtype,
            )
            self.param.append(self.weight)
            
            if bias_shape is not None:
                self.bias = initializer(
                    shape=bias_shape,
                    initializer=self.bias_initializer,
                    dtype=self.dtype,
                )
                self.param.append(self.bias)
            else:
                self.bias = None
            
            if self.trainable==False:
                self.param=[]
            Module.param.extend(self.param)
        ret = tf.einsum(self.equation, data, self.weight)
        if self.bias is not None:
            ret += self.bias
        if self.activation is not None:
            ret = self.activation(ret)
        return ret


def _analyze_einsum_string(equation, bias_axes, input_shape, output_shape):
    """Analyzes an einsum string to determine the required weight shape."""

    dot_replaced_string = re.sub(r"\.\.\.", "0", equation)

    # This is the case where no ellipses are present in the string.
    split_string = re.match(
        "([a-zA-Z]+),([a-zA-Z]+)->([a-zA-Z]+)", dot_replaced_string
    )
    if split_string:
        return _analyze_split_string(
            split_string, bias_axes, input_shape, output_shape
        )

    # This is the case where ellipses are present on the left.
    split_string = re.match(
        "0([a-zA-Z]+),([a-zA-Z]+)->0([a-zA-Z]+)", dot_replaced_string
    )
    if split_string:
        return _analyze_split_string(
            split_string, bias_axes, input_shape, output_shape, left_elided=True
        )

    # This is the case where ellipses are present on the right.
    split_string = re.match(
        "([a-zA-Z]{2,})0,([a-zA-Z]+)->([a-zA-Z]+)0", dot_replaced_string
    )
    if split_string:
        return _analyze_split_string(
            split_string, bias_axes, input_shape, output_shape
        )

    raise ValueError(
        f"Invalid einsum equation '{equation}'. Equations must be in the form "
        "[X],[Y]->[Z], ...[X],[Y]->...[Z], or [X]...,[Y]->[Z]...."
    )


def _analyze_split_string(
    split_string, bias_axes, input_shape, output_shape, left_elided=False
):
    """Analyze an pre-split einsum string to find the weight shape."""
    input_spec = split_string.group(1)
    weight_spec = split_string.group(2)
    output_spec = split_string.group(3)
    elided = len(input_shape) - len(input_spec)

    if isinstance(output_shape, int):
        output_shape = [output_shape]
    else:
        output_shape = list(output_shape)

    output_shape.insert(0, input_shape[0])

    if elided > 0 and left_elided:
        for i in range(1, elided):
            # We already inserted the 0th input dimension at dim 0, so we need
            # to start at location 1 here.
            output_shape.insert(1, input_shape[i])
    elif elided > 0 and not left_elided:
        for i in range(len(input_shape) - elided, len(input_shape)):
            output_shape.append(input_shape[i])

    if left_elided:
        # If we have beginning dimensions elided, we need to use negative
        # indexing to determine where in the input dimension our values are.
        input_dim_map = {
            dim: (i + elided) - len(input_shape)
            for i, dim in enumerate(input_spec)
        }
        # Because we've constructed the full output shape already, we don't need
        # to do negative indexing.
        output_dim_map = {
            dim: (i + elided) for i, dim in enumerate(output_spec)
        }
    else:
        input_dim_map = {dim: i for i, dim in enumerate(input_spec)}
        output_dim_map = {dim: i for i, dim in enumerate(output_spec)}

    for dim in input_spec:
        input_shape_at_dim = input_shape[input_dim_map[dim]]
        if dim in output_dim_map:
            output_shape_at_dim = output_shape[output_dim_map[dim]]
            if (
                output_shape_at_dim is not None
                and output_shape_at_dim != input_shape_at_dim
            ):
                raise ValueError(
                    "Input shape and output shape do not match at shared "
                    f"dimension '{dim}'. Input shape is {input_shape_at_dim}, "
                    "and output shape "
                    f"is {output_shape[output_dim_map[dim]]}."
                )

    for dim in output_spec:
        if dim not in input_spec and dim not in weight_spec:
            raise ValueError(
                f"Dimension '{dim}' was specified in the output "
                f"'{output_spec}' but has no corresponding dim in the input "
                f"spec '{input_spec}' or weight spec '{output_spec}'"
            )

    weight_shape = []
    for dim in weight_spec:
        if dim in input_dim_map:
            weight_shape.append(input_shape[input_dim_map[dim]])
        elif dim in output_dim_map:
            weight_shape.append(output_shape[output_dim_map[dim]])
        else:
            raise ValueError(
                f"Weight dimension '{dim}' did not have a match in either "
                f"the input spec '{input_spec}' or the output "
                f"spec '{output_spec}'. For this layer, the weight must "
                "be fully specified."
            )

    if bias_axes is not None:
        num_left_elided = elided if left_elided else 0
        idx_map = {
            char: output_shape[i + num_left_elided]
            for i, char in enumerate(output_spec)
        }

        for char in bias_axes:
            if char not in output_spec:
                raise ValueError(
                    f"Bias dimension '{char}' was requested, but is not part "
                    f"of the output spec '{output_spec}'"
                )

        first_bias_location = min(
            [output_spec.find(char) for char in bias_axes]
        )
        bias_output_spec = output_spec[first_bias_location:]

        bias_shape = [
            idx_map[char] if char in bias_axes else 1
            for char in bias_output_spec
        ]

        if not left_elided:
            for _ in range(elided):
                bias_shape.append(1)
    else:
        bias_shape = None

    return weight_shape, bias_shape, output_shape
