from Note import nn
from typing import List, Union


def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> Union[int, List[int]]:
    if any([isinstance(v, (tuple, list)) for v in [kernel_size, stride, dilation]]):
        kernel_size, stride, dilation = nn.to_2tuple(kernel_size), nn.to_2tuple(stride), nn.to_2tuple(dilation)
        return [get_padding(*a) for a in zip(kernel_size, stride, dilation)]
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding