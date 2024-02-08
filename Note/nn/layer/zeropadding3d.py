import tensorflow as tf


def normalize_tuple(value, n, allow_zero=False):
    error_msg = (
        f"integers. Received: {value}"
    )

    if isinstance(value, int):
        value_tuple = (value,) * n
    else:
        try:
            value_tuple = tuple(value)
        except TypeError:
            raise ValueError(error_msg)
        if len(value_tuple) != n:
            raise ValueError(error_msg)
        for single_value in value_tuple:
            try:
                int(single_value)
            except (ValueError, TypeError):
                error_msg += (
                    f"including element {single_value} of "
                    f"type {type(single_value)}"
                )
                raise ValueError(error_msg)

    if allow_zero:
        unqualified_values = {v for v in value_tuple if v < 0}
        req_msg = ">= 0"
    else:
        unqualified_values = {v for v in value_tuple if v <= 0}
        req_msg = "> 0"

    if unqualified_values:
        error_msg += (
            f" including {unqualified_values}"
            f" that does not satisfy the requirement `{req_msg}`."
        )
        raise ValueError(error_msg)

    return value_tuple


class zeropadding3d:
    def __init__(self,input_size=None, padding=(1, 1, 1)):
        self.pattern = None
        if padding is not None:
            if isinstance(padding, int):
                padding = (
                    (padding, padding),
                    (padding, padding),
                    (padding, padding),
                )
            else:
                if len(padding) != 3:
                    raise ValueError(
                        f"`padding` should have 3 elements. Received: {padding}."
                    )
                dim1_padding = normalize_tuple(
                    padding[0], 2, "1st entry of padding", allow_zero=True
                )
                dim2_padding = normalize_tuple(
                    padding[1], 2, "2nd entry of padding", allow_zero=True
                )
                dim3_padding = normalize_tuple(
                    padding[2], 2, "3rd entry of padding", allow_zero=True
                )
                padding = (dim1_padding, dim2_padding, dim3_padding)
            self.pattern = [
                [0, 0],
                [padding[0][0], padding[0][1]],
                [padding[1][0], padding[1][1]],
                [padding[2][0], padding[2][1]],
                [0, 0],
            ]
            
        self.input_size=input_size
        if input_size!=None:
            self.output_size=input_size
            
            
    def __call__(self, data, padding=(1, 1, 1)):
        if self.pattern is None:
            if isinstance(padding, int):
                padding = (
                    (padding, padding),
                    (padding, padding),
                    (padding, padding),
                )
            else:
                if len(padding) != 3:
                    raise ValueError(
                        f"`padding` should have 3 elements. Received: {padding}."
                    )
                dim1_padding = normalize_tuple(
                    padding[0], 2, "1st entry of padding", allow_zero=True
                )
                dim2_padding = normalize_tuple(
                    padding[1], 2, "2nd entry of padding", allow_zero=True
                )
                dim3_padding = normalize_tuple(
                    padding[2], 2, "3rd entry of padding", allow_zero=True
                )
                padding = (dim1_padding, dim2_padding, dim3_padding)
            pattern = [
                [0, 0],
                [padding[0][0], padding[0][1]],
                [padding[1][0], padding[1][1]],
                [padding[2][0], padding[2][1]],
                [0, 0],
            ]
        else:
            pattern = self.pattern
        return tf.pad(data, pattern)